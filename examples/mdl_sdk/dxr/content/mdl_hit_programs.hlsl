/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#if !defined(TARGET_CODE_ID)
    #define TARGET_CODE_ID 0
#endif

// macros to append the target code ID to the function name.
// this is required because the resulting DXIL libraries will be linked to same pipeline object
// and for that, the entry point names have to be unique.
#define export_name_impl(name, id) name ## _ ## id
#define export_name_impl_2(name, id) export_name_impl(name, id)
#define export_name(name) export_name_impl_2(name, TARGET_CODE_ID)

#define MDL_RADIANCE_ANY_HIT_PROGRAM        export_name(MdlRadianceAnyHitProgram)
#define MDL_RADIANCE_CLOSEST_HIT_PROGRAM    export_name(MdlRadianceClosestHitProgram)
#define MDL_SHADOW_ANY_HIT_PROGRAM          export_name(MdlShadowAnyHitProgram)

#include "content/environment.hlsl"

// ------------------------------------------------------------------------------------------------
// Local Root Signature
// ------------------------------------------------------------------------------------------------

// when the hit programm has access to the SBT we can use the data directly
// otherwise we would have the access them them through the heap using:

// #define FEATURE_HIT_PROGRAMM_SBT_ACCESS 1
#if (FEATURE_HIT_PROGRAMM_SBT_ACCESS == 1)
    ByteAddressBuffer sbt_vertices : register(t1, space0);
    StructuredBuffer<uint> sbt_indices: register(t2, space0);
#endif


// geometry data
// as long as there are only a few values here, place them directly instead of a constant buffer
cbuffer Local_Constants_b0_space1 : register(b0, space1) { uint geomerty_mesh_resource_heap_index; }
cbuffer Local_Constants_b1_space1 : register(b1, space1) { uint geometry_instance_resource_heap_index; }
cbuffer Local_Constants_b2_space1 : register(b2, space1) { uint geometry_part_vertex_buffer_byte_offset; }
cbuffer Local_Constants_b3_space1 : register(b3, space1) { uint geometry_part_vertex_stride; }
cbuffer Local_Constants_b4_space1 : register(b4, space1) { uint geometry_part_index_offset; }
cbuffer Local_Constants_b5_space1 : register(b5, space1) { uint geometry_part_scene_data_info_offset; }
cbuffer Local_Constants_b6_space1 : register(b6, space1) { uint material_target_heap_index; }
cbuffer Local_Constants_b7_space1 : register(b7, space1) { uint material_instance_heap_index; }

// ------------------------------------------------------------------------------------------------
// entrypoint -> runtime interface
// entrypoint -> renderer
// ------------------------------------------------------------------------------------------------

uint get_ro_data_segment_heap_index() { return material_target_heap_index; }
ConstantBuffer<Material_constants> get_current_material()
{
    #if (FEATURE_DYNAMIC_RESOURCES == 1)
        return ResourceDescriptorHeap[NonUniformResourceIndex(material_instance_heap_index)];
    #else
        return Global_CBVs_Material_constants[NonUniformResourceIndex(material_instance_heap_index)];
    #endif
}
uint get_argument_block_heap_index() { return material_instance_heap_index + 1; }
uint get_texture_infos_heap_index() { return material_instance_heap_index + 2; }
uint get_light_profile_heap_index() { return material_instance_heap_index + 3; }
uint get_mbsdf_infos_heap_index() { return material_instance_heap_index + 4; }
uint get_vertex_buffer_heap_index() { return geomerty_mesh_resource_heap_index; }
uint get_index_buffer_heap_index() { return geomerty_mesh_resource_heap_index + 1; }
uint get_scene_data_info_heap_index() { return geometry_instance_resource_heap_index; }
uint get_scene_data_buffer_heap_index() { return geometry_instance_resource_heap_index + 1; }


// ------------------------------------------------------------------------------------------------
// helper
// ------------------------------------------------------------------------------------------------

// selects one light source randomly
float3 sample_lights(
    Shading_state_material state, out float3 to_light, out float light_pdf, inout uint seed)
{
    float p_select_light = 1.0f;
    if (scene_constants.point_light_enabled != 0)
    {
        // keep it simple and use either point light or environment light, each with the same
        // probability. If the environment factor is zero, we always use the point light
        // Note: see also miss shader
        p_select_light = scene_constants.environment_intensity_factor > 0.0f ? 0.5f : 1.0f;

        // in general, you would select the light depending on the importance of it
        // e.g. by incorporating their luminance

        // randomly select one of the lights
        if (rnd(seed) <= p_select_light)
        {
            light_pdf = DIRAC; // infinity

            // compute light direction and distance
            #if defined(USE_DERIVS)
                to_light = scene_constants.point_light_position - state.position.val;
            #else
                to_light = scene_constants.point_light_position - state.position;
            #endif
            const float inv_distance2 = 1.0f / dot(to_light, to_light);
            to_light *= sqrt(inv_distance2);

            return scene_constants.point_light_intensity *
                inv_distance2 * 0.25f * M_ONE_OVER_PI / p_select_light;
        }

        // probability to select the environment instead
        p_select_light = (1.0f - p_select_light);
    }

    // light from the environment
    float3 radiance = environment_sample(   // (see common.hlsl)
        seed,
        to_light,
        light_pdf);

    // return radiance over pdf
    light_pdf *= p_select_light;
    return radiance / light_pdf;
}


ByteAddressBuffer get_vertex_buffer()
{
    #if (FEATURE_HIT_PROGRAMM_SBT_ACCESS == 1)
        // fastest (direct access because the address is in the root signature)
        return sbt_vertices;
    #elif (FEATURE_DYNAMIC_RESOURCES == 1)
        // slower but more flexible (one indirection though the heap)
        return ResourceDescriptorHeap[NonUniformResourceIndex(get_vertex_buffer_heap_index())];
    #else
        // even slower but also flexible (indirection through the heap and a descriptor table)
        return Global_SRVs_ByteAddressBuffer[NonUniformResourceIndex(
            get_vertex_buffer_heap_index())];
    #endif
}

StructuredBuffer<uint> get_index_buffer()
{
    #if (FEATURE_HIT_PROGRAMM_SBT_ACCESS == 1)
        return sbt_indices;
    #elif (FEATURE_DYNAMIC_RESOURCES == 1)
        // slower but more flexible (one indirection though the heap)
        return ResourceDescriptorHeap[NonUniformResourceIndex(get_index_buffer_heap_index())];
    #else
        // even slower but also flexible (indirection through the heap and a descriptor table)
        return Global_SRVs_StructuredBuffer_uint[NonUniformResourceIndex(
            get_index_buffer_heap_index())];
    #endif
}

// fetch vertex data with known layout
float3 fetch_vertex_data_float3(ByteAddressBuffer vb, const uint index, const uint byte_offset)
{
    const uint address =
        geometry_part_vertex_buffer_byte_offset + // base address for this part of the mesh
        geometry_part_vertex_stride * index +     // offset to the selected vertex
        byte_offset;                         // offset within the vertex

    // mesh data, includes the per mesh scene data
    return asfloat(vb.Load3(address));
}

// fetch vertex data with known layout
float4 fetch_vertex_data_float4(ByteAddressBuffer vb, const uint index, const uint byte_offset)
{
    const uint address =
        geometry_part_vertex_buffer_byte_offset + // base address for this part of the mesh
        geometry_part_vertex_stride * index +     // offset to the selected vertex
        byte_offset;                         // offset within the vertex

    return asfloat(vb.Load4(address));
}

bool is_back_face()
{
    // get vertex and index buffer
    ByteAddressBuffer vb = get_vertex_buffer();
    StructuredBuffer<uint> ib = get_index_buffer();

    // get vertex indices for the hit triangle
    const uint index_offset = 3 * PrimitiveIndex() + geometry_part_index_offset;
    const uint3 vertex_indices = uint3(
        ib[index_offset + 0], ib[index_offset + 1], ib[index_offset + 2]);

    // get position of the hit point
    const float3 pos0 = fetch_vertex_data_float3(vb, vertex_indices.x, VERT_BYTEOFFSET_POSITION);
    const float3 pos1 = fetch_vertex_data_float3(vb, vertex_indices.y, VERT_BYTEOFFSET_POSITION);
    const float3 pos2 = fetch_vertex_data_float3(vb, vertex_indices.z, VERT_BYTEOFFSET_POSITION);

    // compute geometry normal and check for back face hit
    const float3 geom_normal = normalize(cross(pos1 - pos0, pos2 - pos0));
    return dot(geom_normal, ObjectRayDirection()) > 0.0f;
}

void setup_mdl_shading_state(
    out Shading_state_material mdl_state,
    Attributes attrib)
{
    // get vertex and index buffer
    ByteAddressBuffer vb = get_vertex_buffer();
    StructuredBuffer<uint> ib = get_index_buffer();

    // get vertex indices for the hit triangle
    const uint index_offset = 3 * PrimitiveIndex() + geometry_part_index_offset;
    const uint3 vertex_indices = uint3(
        ib[index_offset + 0], ib[index_offset + 1], ib[index_offset + 2]);

    // coordinates inside the triangle
    const float3 barycentric = float3(
        1.0f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    // mesh transformations
    const float4x4 object_to_world = to4x4(ObjectToWorld());
    const float4x4 world_to_object = to4x4(WorldToObject());

    // get position of the hit point
    const float3 pos0 = fetch_vertex_data_float3(vb, vertex_indices.x, VERT_BYTEOFFSET_POSITION);
    const float3 pos1 = fetch_vertex_data_float3(vb, vertex_indices.y, VERT_BYTEOFFSET_POSITION);
    const float3 pos2 = fetch_vertex_data_float3(vb, vertex_indices.z, VERT_BYTEOFFSET_POSITION);
    float3 hit_position = pos0 * barycentric.x + pos1 * barycentric.y + pos2 * barycentric.z;
    hit_position = mul(object_to_world, float4(hit_position, 1)).xyz;

    // get normals (geometry normal and interpolated vertex normal)
    const float3 geom_normal = normalize(cross(pos1 - pos0, pos2 - pos0));
    const float3 normal = normalize(
        fetch_vertex_data_float3(vb, vertex_indices.x, VERT_BYTEOFFSET_NORMAL) * barycentric.x +
        fetch_vertex_data_float3(vb, vertex_indices.y, VERT_BYTEOFFSET_NORMAL) * barycentric.y +
        fetch_vertex_data_float3(vb, vertex_indices.z, VERT_BYTEOFFSET_NORMAL) * barycentric.z);

    // transform normals using inverse transpose
    // -  world_to_object = object_to_world^-1
    // -  mul(v, world_to_object) = mul(object_to_world^-T, v)
    float3 world_geom_normal = normalize(mul(float4(geom_normal, 0), world_to_object).xyz);
    float3 world_normal = normalize(mul(float4(normal, 0), world_to_object).xyz);

    // reconstruct tangent frame from vertex data
    float3 world_tangent, world_binormal;
    float4 tangent0 =
        fetch_vertex_data_float4(vb, vertex_indices.x, VERT_BYTEOFFSET_TANGENT) * barycentric.x +
        fetch_vertex_data_float4(vb, vertex_indices.y, VERT_BYTEOFFSET_TANGENT) * barycentric.y +
        fetch_vertex_data_float4(vb, vertex_indices.z, VERT_BYTEOFFSET_TANGENT) * barycentric.z;
    tangent0.xyz = normalize(tangent0.xyz);
    world_tangent = normalize(mul(object_to_world, float4(tangent0.xyz, 0)).xyz);
    world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
    world_binormal = cross(world_normal, world_tangent) * tangent0.w;

    // flip normals to the side of the incident ray
    const bool backfacing_primitive = dot(world_geom_normal, WorldRayDirection()) > 0.0;
    if (backfacing_primitive)
        world_geom_normal *= -1.0f;

    if (dot(world_normal, WorldRayDirection()) > 0.0)
        world_normal *= -1.0f;

    // handle low tessellated meshes with smooth normals
    float3 k2 = reflect(WorldRayDirection(), world_normal);
    if (dot(world_geom_normal, k2) < 0.0f)
        world_normal = world_geom_normal;

    // fill the actual state fields used by MD
    mdl_state.normal = world_normal;
    mdl_state.geom_normal = world_geom_normal;
    #if defined(USE_DERIVS)
        // currently not supported
        mdl_state.position.val = hit_position;
        mdl_state.position.dx = float3(0, 0, 0);
        mdl_state.position.dy = float3(0, 0, 0);
    #else
        mdl_state.position = hit_position;
    #endif
    mdl_state.animation_time = scene_constants.enable_animiation ? scene_constants.total_time : 0.0f;
    mdl_state.tangent_u[0] = world_tangent;
    mdl_state.tangent_v[0] = world_binormal;
    // #if defined(USE_TEXTURE_RESULTS)
    // filling the buffer with zeros not required
    //     mdl_state.text_results = (float4[MDL_NUM_TEXTURE_RESULTS]) 0;
    // #endif
    mdl_state.ro_data_segment_offset = 0;
    mdl_state.world_to_object = world_to_object;
    mdl_state.object_to_world = object_to_world;
    mdl_state.object_id = 0;
    mdl_state.meters_per_scene_unit = scene_constants.meters_per_scene_unit;
    mdl_state.arg_block_offset = 0;

    // fill the renderer state information
    mdl_state.renderer_state.scene_data_info_offset = geometry_part_scene_data_info_offset;
    mdl_state.renderer_state.scene_data_geometry_byte_offset = geometry_part_vertex_buffer_byte_offset;
    mdl_state.renderer_state.hit_vertex_indices = vertex_indices;
    mdl_state.renderer_state.barycentric = barycentric;
    mdl_state.renderer_state.hit_backface = backfacing_primitive;

    // get texture coordinates using a manually added scene data element with the scene data id
    // defined as `SCENE_DATA_ID_TEXCOORD_0`
    // (see end of target code generation on application side)
    float2 texcoord0 = scene_data_lookup_float2(
        mdl_state, SCENE_DATA_ID_TEXCOORD_0, float2(0.0f, 0.0f), false);

    // apply uv transformations
    texcoord0 = texcoord0 * scene_constants.uv_scale + scene_constants.uv_offset;
    if (scene_constants.uv_repeat != 0)
    {
        texcoord0 = texcoord0 - floor(texcoord0);
    }
    if (scene_constants.uv_saturate != 0)
    {
        texcoord0 = saturate(texcoord0);
    }

    #if defined(USE_DERIVS)
        // would make sense in a rasterizer. for a ray tracers this is not straight forward
        mdl_state.text_coords[0].val = float3(texcoord0, 0);
        mdl_state.text_coords[0].dx = float3(0, 0, 0); // float3(ddx(texcoord0), 0);
        mdl_state.text_coords[0].dy = float3(0, 0, 0); // float3(ddy(texcoord0), 0);
    #else
        mdl_state.text_coords[0] = float3(texcoord0, 0);
    #endif
}


// ------------------------------------------------------------------------------------------------
// MDL hit group shader
// ------------------------------------------------------------------------------------------------

[shader("anyhit")]
void MDL_RADIANCE_ANY_HIT_PROGRAM(inout RadianceHitInfo payload, Attributes attrib)
{
    ConstantBuffer<Material_constants> mat = get_current_material();

    // back face culling
    if (mat.is_single_sided() && is_back_face())
    {
        IgnoreHit();
        return;
    }

    // early out if there is no opacity function, it's a hit.
    if (!mat.has_cutout_opacity())
        return;

    // setup MDL state
    Shading_state_material mdl_state;
    setup_mdl_shading_state(mdl_state, attrib);

    // evaluate the cutout opacity
    const float opacity = mdl_standalone_geometry_cutout_opacity(mdl_state);

    // do alpha blending the stochastically way
    if (rnd(payload.seed) < opacity)
        return;

    IgnoreHit();
}

[shader("closesthit")]
void MDL_RADIANCE_CLOSEST_HIT_PROGRAM(inout RadianceHitInfo payload, Attributes attrib)
{
    ConstantBuffer<Material_constants> mat = get_current_material();

    // setup MDL state
    Shading_state_material mdl_state;
    setup_mdl_shading_state(mdl_state, attrib);

    // pre-compute and cache data used by different generated MDL functions
    if (mat.has_init())
    {
        mdl_init(mdl_state);
    }

    // Illustrate the usage of AOVs.
    // A custom material type is probably very well known to the renderer or other components that use the code.
    // Here, we simply display AOVs as color rather than feeding the results into a simulation or
    // post processing pipeline for example.
    if (mat.has_aovs() && scene_constants.aov_index_to_render >= 0)
    {
        float3 aov_as_color = mdl_aov(scene_constants.aov_index_to_render, mdl_state);
        payload.contribution = aov_as_color;
        add_flag(payload.flags, FLAG_DONE);
        return;
    }

    // thin-walled materials are allowed to have a different back side
    // buy the can't have volumetric properties
    const bool thin_walled = mat.can_be_thin_walled() ? mdl_thin_walled(mdl_state) : false;

    // for thin-walled materials there is no 'inside'
    const bool inside = has_flag(payload.flags, FLAG_INSIDE);
    const float ior1 = (inside &&!thin_walled) ? BSDF_USE_MATERIAL_IOR : 1.0f;
    const float ior2 = (inside &&!thin_walled) ? 1.0f : BSDF_USE_MATERIAL_IOR;


    // apply volume attenuation
    //---------------------------------------------------------------------------------------------
    if (inside && !thin_walled && (mat.has_volume_absorption() || mat.has_volume_scattering())) 
    {
        const float3 a_coeff = mdl_volume_absorption_coefficient(mdl_state);
        const float3 s_coeff = mdl_volume_scattering_coefficient(mdl_state);
        const float3 t_coeff = a_coeff + s_coeff;
        const float g = mdl_volume_scattering_directional_bias(mdl_state);

        // distance the ray traveled in meters
#if defined(USE_DERIVS)
        float distance = length(mdl_state.position.val - payload.ray_origin_next) * scene_constants.meters_per_scene_unit;
#else
        float distance = length(mdl_state.position - payload.ray_origin_next) * scene_constants.meters_per_scene_unit;
#endif

        // scatter only if we have non-zero scatter coefficients
        float survival_prob = 1.0f;
        if (s_coeff.x > 0.0f || s_coeff.y > 0.0 || s_coeff.z > 0.0f)
        {
            const float s_coeff_max = max(s_coeff.x, max(s_coeff.y, s_coeff.z));
            const float t_coeff_min = min(t_coeff.x, min(t_coeff.y, t_coeff.z));

            float sample_coeff = 0.0f;
            if (s_coeff_max <= t_coeff_min)
            {
                // can use common coefficient for distance importance sampling while keeping variance bounded
                sample_coeff = t_coeff_min;
            }
            else if ((payload.flags & (FLAG_SSS_R | FLAG_SSS_G | FLAG_SSS_B)) == 0)
            {
                // switch to single color
                const float xi = rnd(payload.seed);
                if (xi < (1.0 / 3.0))
                {
                    payload.flags |= FLAG_SSS_R;
                    payload.weight.x *= 3.0f;
                    payload.weight.y = 0.0f;
                    payload.weight.z = 0.0f;
                }
                else if (xi < (2.0 / 3.0))
                {
                    payload.flags |= FLAG_SSS_G;
                    payload.weight.x = 0.0f;
                    payload.weight.y *= 3.0f;
                    payload.weight.z = 0.0f;
                }
                else
                {
                    payload.flags |= FLAG_SSS_B;
                    payload.weight.x = 0.0f;
                    payload.weight.y = 0.0f;
                    payload.weight.z *= 3.0f;
                }
            }

            if (has_flag(payload.flags, FLAG_SSS_R))
                sample_coeff = t_coeff.x;
            else if (payload.flags & FLAG_SSS_G)
                sample_coeff = t_coeff.y;
            else if (payload.flags & FLAG_SSS_B)
                sample_coeff = t_coeff.z;

            // sample travel distance in meters
            const float sampled_distance = -log(1.0 - rnd(payload.seed)) / sample_coeff;

            // scattering event happened
            if (sampled_distance < distance)
            {
                // compute scattering position in scene units
                distance = sampled_distance;
                payload.ray_origin_next += payload.ray_direction_next * (distance / scene_constants.meters_per_scene_unit);

                // random direction, Henyey-Greenstein phase function
                float cosTheta;
                if (g < 0.001)
                    cosTheta = 1.0 - 2.0 * rnd(payload.seed);
                else
                {
                    const float inner_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * rnd(payload.seed));
                    cosTheta = (1.0 + g * g - inner_term * inner_term) / (2.0 * g);
                }
                const float phi = 2.0 * M_PI * rnd(payload.seed);
                const float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
                float sinPhi, cosPhi;
                sincos(phi, sinPhi, cosPhi);

                float3 u, v;
                create_basis(payload.ray_direction_next, u, v);
                payload.ray_direction_next = u * cosPhi * sinTheta + v * sinPhi * sinTheta + payload.ray_direction_next * cosTheta;

                // apply scattering and volume attenuation
                // note, this is limited to uniform scattering for subsurface scattering
                // for non-uniform volume data, this is not sufficient
                float pdf_dist = sample_coeff * exp(-sample_coeff * distance);
                if (payload.weight.x > 0.0f)
                    payload.weight.x *= s_coeff.x * exp(-t_coeff.x * distance) / pdf_dist;
                if (payload.weight.y > 0.0f)
                    payload.weight.y *= s_coeff.y * exp(-t_coeff.y * distance) / pdf_dist;
                if (payload.weight.z > 0.0f)
                    payload.weight.z *= s_coeff.z * exp(-t_coeff.z * distance) / pdf_dist;

                // continue the ray into scattering direction
                add_flag(payload.flags, FLAG_SSS); // for counting SSS steps separate
                return;
            }
            remove_flag(payload.flags, FLAG_SSS); // for counting SSS steps separate

            // probability to reach the distance
            survival_prob = exp(-sample_coeff * distance);
        }

        // apply volume attenuation
        if (payload.weight.x > 0.0f)
            payload.weight.x *= exp(-t_coeff.x * distance) / survival_prob;
        if (payload.weight.y > 0.0f)
            payload.weight.y *= exp(-t_coeff.y * distance) / survival_prob;
        if (payload.weight.z > 0.0f)
            payload.weight.z *= exp(-t_coeff.z * distance) / survival_prob;
    }

    // add emission
    //---------------------------------------------------------------------------------------------
    const bool has_surface_emission = mat.has_surface_emission();
    const bool has_backface_emission = mat.has_backface_emission();
    if (has_surface_emission || has_backface_emission)
    {
        // evaluate EDF
        Edf_evaluate_data eval_data = (Edf_evaluate_data) 0;
        eval_data.k1 = -WorldRayDirection();

        // evaluate intensity expression
        float3 intensity = float3(0.0f, 0.0f, 0.0f);
        if (has_backface_emission && thin_walled && mdl_state.renderer_state.hit_backface)
            intensity = mdl_backface_emission_intensity(mdl_state);
        else if (has_surface_emission)
            intensity = mdl_surface_emission_intensity(mdl_state);

        #if (MDL_DF_HANDLE_SLOT_MODE == -1)

            // evaluate the distribution function
            if (has_backface_emission && thin_walled && mdl_state.renderer_state.hit_backface)
            {
                mdl_backface_emission_evaluate(eval_data, mdl_state);
            }
            else if (has_surface_emission)
            {
                mdl_surface_emission_evaluate(eval_data, mdl_state);
            }
        
            // add emission
            payload.contribution += payload.weight * intensity * eval_data.edf;

        #else
            for(uint offset = 0; offset < MDL_DF_HANDLE_SLOT_COUNT; offset += MDL_DF_HANDLE_SLOT_MODE)
            {
                // evaluate the distribution function
                eval_data.handle_offset = offset;
                if (has_backface_emission && thin_walled && mdl_state.renderer_state.hit_backface)
                {
                    mdl_backface_emission_evaluate(eval_data, mdl_state);
                }
                else if (has_surface_emission)
                {
                    mdl_surface_emission_evaluate(eval_data, mdl_state);
                }
        
                // add emission
                for (uint lobe = 0; lobe < MDL_DF_HANDLE_SLOT_MODE; ++lobe)
                    payload.contribution += payload.weight * intensity * eval_data.edf[lobe];
            }
#endif
    }

    // Write Auxiliary Buffers
    //---------------------------------------------------------------------------------------------
    const bool has_surface_scattering = mat.has_surface_scattering();
    const bool has_backface_scattering = mat.has_backface_scattering();
    #if defined(ENABLE_AUXILIARY)
    if (has_flag(payload.flags, FLAG_FIRST_PATH_SEGMENT))
    {
        Bsdf_auxiliary_data aux_data = (Bsdf_auxiliary_data) 0;
        aux_data.ior1 = ior1;                    // IOR current medium
        aux_data.ior2 = ior2;                    // IOR other side
        aux_data.k1 = -WorldRayDirection();      // outgoing direction
        aux_data.flags = scene_constants.bsdf_data_flags;
        uint3 launch_index =  DispatchRaysIndex();

        #if (FEATURE_DYNAMIC_RESOURCES == 1)
            RWTexture2D<float4> AlbedoDiffuseBuffer = ResourceDescriptorHeap[2];
            RWTexture2D<float4> AlbedoGlossyBuffer = ResourceDescriptorHeap[3];
            RWTexture2D<float4> NormalBuffer = ResourceDescriptorHeap[4];
            RWTexture2D<float4> RoughnessBuffer = ResourceDescriptorHeap[5];
        #else
            RWTexture2D<float4> AlbedoDiffuseBuffer = Global_UAVs_Texture2D_float4[2];
            RWTexture2D<float4> AlbedoGlossyBuffer = Global_UAVs_Texture2D_float4[3];
            RWTexture2D<float4> NormalBuffer = Global_UAVs_Texture2D_float4[4];
            RWTexture2D<float4> RoughnessBuffer = Global_UAVs_Texture2D_float4[5];
        #endif

        #if (MDL_DF_HANDLE_SLOT_MODE == -1)

            if (has_backface_scattering && thin_walled && mdl_state.renderer_state.hit_backface)
            {
                mdl_backface_scattering_auxiliary(aux_data, mdl_state);
            }
            else if (has_surface_scattering)
            {
                mdl_surface_scattering_auxiliary(aux_data, mdl_state);
            }
        
            AlbedoDiffuseBuffer[launch_index.xy] = float4(aux_data.albedo_diffuse, 1.0f);
            AlbedoGlossyBuffer[launch_index.xy] = float4(aux_data.albedo_glossy, 1.0f);
            NormalBuffer[launch_index.xy] = float4(aux_data.normal, 1.0f);
            RoughnessBuffer[launch_index.xy] = float4(aux_data.roughness.xy, 0.0f, 1.0f);

        #else
            float3 aux_albedo_diffuse = float3(0.0f, 0.0f, 0.0f);
            float3 aux_albedo_glossy = float3(0.0f, 0.0f, 0.0f);
            float3 aux_normal = float3(0.0f, 0.0f, 0.0f);
            float aux_roughness_weight_sum = 0.0f;
            float2 aux_rouhness = float2(0.0f, 0.0f);
            for(uint offset = 0; offset < MDL_DF_HANDLE_SLOT_COUNT; offset += MDL_DF_HANDLE_SLOT_MODE)
            {
                aux_data.handle_offset = offset;
                if (has_backface_scattering && thin_walled && mdl_state.renderer_state.hit_backface)
                {
                    mdl_backface_scattering_auxiliary(aux_data, mdl_state);
                }
                else if (has_surface_scattering)
                {
                    mdl_surface_scattering_auxiliary(aux_data, mdl_state);
                }

                for (uint lobe = 0; lobe < MDL_DF_HANDLE_SLOT_MODE; ++lobe)
                {
                    aux_albedo_diffuse += aux_data.albedo_diffuse[lobe];
                    aux_albedo_glossy += aux_data.albedo_glossy[lobe];
                    aux_normal += aux_data.normal[lobe];
                    aux_rouhness += aux_data.roughness[lobe].xy * aux_data.roughness[lobe].z;
                    aux_roughness_weight_sum += aux_data.roughness[lobe].z;
                }
            }
            AlbedoDiffuseBuffer[launch_index.xy] = float4(aux_albedo_diffuse, 0.0f);
            AlbedoGlossyBuffer[launch_index.xy] = float4(aux_albedo_glossy, 0.0f);
            NormalBuffer[launch_index.xy] = float4(aux_normal, 0.0f);
            RoughnessBuffer[launch_index.xy] = float4(aux_rouhness / aux_roughness_weight_sum, 0.0f, 0.0f);
        #endif
    }
    #endif

    // Sample Light Sources for next event estimation
    //---------------------------------------------------------------------------------------------

    float3 to_light;
    float light_pdf;
    float3 radiance_over_pdf = sample_lights(mdl_state, to_light, light_pdf, payload.seed);

    // do not next event estimation (but delay the adding of contribution)
    float3 contribution = float3(0.0f, 0.0f, 0.0f);
    const bool next_event_valid = ((dot(to_light, mdl_state.geom_normal) > 0.0f) != inside) && light_pdf != 0.0f &&
        !has_flag(payload.flags, FLAG_LAST_PATH_SEGMENT);

    if (next_event_valid)
    {
        // call generated mdl function to evaluate the scattering BSDF
        Bsdf_evaluate_data eval_data = (Bsdf_evaluate_data)0;
        eval_data.ior1 = ior1;
        eval_data.ior2 = ior2;
        eval_data.k1 = -WorldRayDirection();
        eval_data.k2 = to_light;
        eval_data.flags = scene_constants.bsdf_data_flags;

        #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        // begin handle loop
        for(uint offset = 0; offset < MDL_DF_HANDLE_SLOT_COUNT; offset += MDL_DF_HANDLE_SLOT_MODE)
        {
            eval_data.handle_offset = offset;
        #endif

            // use backface instead of surface scattering?
            if (has_backface_scattering && thin_walled && mdl_state.renderer_state.hit_backface)
            {
                mdl_backface_scattering_evaluate(eval_data, mdl_state);
            }
            else if (has_surface_scattering)
            {
                mdl_surface_scattering_evaluate(eval_data, mdl_state);
            }

            // compute lighting for this light
            if(eval_data.pdf > 0.0f)
            {
                const float mis_weight = (light_pdf == DIRAC)
                    ? 1.0f
                    : light_pdf / (light_pdf + eval_data.pdf);

                // sample weight
                const float3 w = payload.weight * radiance_over_pdf * mis_weight;
                #if (MDL_DF_HANDLE_SLOT_MODE == -1)
                    contribution += w * eval_data.bsdf_diffuse;
                    contribution += w * eval_data.bsdf_glossy;
                #else
                    for (uint i = 0; i < MDL_DF_HANDLE_SLOT_MODE; ++i)
                    {
                        contribution += w * eval_data.bsdf_diffuse[i];
                        contribution += w * eval_data.bsdf_glossy[i];
                    }
                #endif
            }

        #if (MDL_DF_HANDLE_SLOT_MODE != -1)
        // end handle loop
        }
        #endif
    }

    // Sample direction of the next ray
    //---------------------------------------------------------------------------------------------

    // not a camera ray anymore
    remove_flag(payload.flags, FLAG_CAMERA_RAY);

    Bsdf_sample_data sample_data = (Bsdf_sample_data) 0;
    sample_data.ior1 = ior1;                    // IOR current medium
    sample_data.ior2 = ior2;                    // IOR other side
    sample_data.k1 = -WorldRayDirection();      // outgoing direction
    sample_data.xi = rnd4(payload.seed);        // random sample number
    sample_data.flags = scene_constants.bsdf_data_flags;

    // use backface instead of surface scattering?
    if (has_backface_scattering && thin_walled && mdl_state.renderer_state.hit_backface)
    {
        mdl_backface_scattering_sample(sample_data, mdl_state);
    }
    else if (has_surface_scattering)
    {
        mdl_surface_scattering_sample(sample_data, mdl_state);
    }

    // stop on absorb
    if (sample_data.event_type == BSDF_EVENT_ABSORB)
    {
        add_flag(payload.flags, FLAG_DONE);
        // no not return here, we need to do next event estimation first
    }
    else
    {
        // flip inside/outside on transmission
        // setup next path segment
        payload.ray_direction_next = sample_data.k2;
        payload.weight *= sample_data.bsdf_over_pdf;
        if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0)
        {
            toggle_flag(payload.flags, FLAG_INSIDE);
            // continue on the opposite side
            #if defined(USE_DERIVS)
                payload.ray_origin_next = offset_ray(mdl_state.position.val, -mdl_state.geom_normal);
            #else
                payload.ray_origin_next = offset_ray(mdl_state.position, -mdl_state.geom_normal);
            #endif
        }
        else
        {
            // continue on the current side
            #if defined(USE_DERIVS)
                payload.ray_origin_next = offset_ray(mdl_state.position.val, mdl_state.geom_normal);
            #else
                payload.ray_origin_next = offset_ray(mdl_state.position, mdl_state.geom_normal);
            #endif
        }

        if ((sample_data.event_type & BSDF_EVENT_SPECULAR) != 0)
            payload.last_bsdf_pdf = DIRAC;
        else
            payload.last_bsdf_pdf = sample_data.pdf;
    }

    // Add contribution from next event estimation if not shadowed
    //---------------------------------------------------------------------------------------------

    // cast a shadow ray; assuming light is always outside
    RayDesc ray;
    #if defined(USE_DERIVS)
        ray.Origin = offset_ray(mdl_state.position.val, mdl_state.geom_normal * (inside ? -1.0f : 1.0f));
    #else
        ray.Origin = offset_ray(mdl_state.position, mdl_state.geom_normal * (inside ? -1.0f : 1.0f));
    #endif
    ray.Direction = to_light;
    ray.TMin = 0.0f;
    ray.TMax = scene_constants.far_plane_distance;

    // prepare the ray and payload but trace at the end to reduce the amount of data that has
    // to be recovered after coming back from the shadow trace
    ShadowHitInfo shadow_payload;
    shadow_payload.isHit = false;
    shadow_payload.seed = payload.seed;

    // Ray tracing acceleration structure
    TraceRay(
        SceneBVH,               // AccelerationStructure
        RAY_FLAG_NONE,          // RayFlags
        0xFF /* allow all */,   // InstanceInclusionMask
        RAY_TYPE_SHADOW,        // RayContributionToHitGroupIndex
        RAY_TYPE_COUNT,         // MultiplierForGeometryContributionToHitGroupIndex
        RAY_TYPE_SHADOW,        // MissShaderIndex
        ray,
        shadow_payload);

    // add to ray contribution from next event estimation
    if (!shadow_payload.isHit)
        payload.contribution += contribution;
}

// ------------------------------------------------------------------------------------------------
// MDL shadow group shader
// ------------------------------------------------------------------------------------------------

[shader("anyhit")]
void MDL_SHADOW_ANY_HIT_PROGRAM(inout ShadowHitInfo payload, Attributes attrib)
{
    ConstantBuffer<Material_constants> mat = get_current_material();

    // back face culling
    if (mat.is_single_sided() && is_back_face())
    {
        IgnoreHit();
        return;
    }

    // early out if there is no opacity function
    if (!mat.has_cutout_opacity())
    {
        payload.isHit = true;
        AcceptHitAndEndSearch();
        return;
    }

    // setup MDL state
    Shading_state_material mdl_state;
    setup_mdl_shading_state(mdl_state, attrib);

    // evaluate the cutout opacity
    const float opacity = mdl_standalone_geometry_cutout_opacity(mdl_state);

    // do alpha blending the stochastically way
    if (rnd(payload.seed) < opacity)
    {
        payload.isHit = true;
        AcceptHitAndEndSearch();
        return;
    }

    IgnoreHit();
}
