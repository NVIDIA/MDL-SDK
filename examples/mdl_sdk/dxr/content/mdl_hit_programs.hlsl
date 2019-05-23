#include "common.hlsl"


enum MaterialFlags
{
    MATERIAL_FLAG_NONE          = 0,
    MATERIAL_FLAG_OPAQUE        = 1 << 0, // allows to skip opacity evaluation
    MATERIAL_FLAG_SINGLE_SIDED  = 1 << 1  // geometry is only visible from the front side
};

// ------------------------------------------------------------------------------------------------
// defined in the global root signature
// ------------------------------------------------------------------------------------------------

// Ray tracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

// Environment map and sample data for importance sampling
Texture2D<float4> environment_texture : register(t0,space1);
StructuredBuffer<Environment_sample_data> environment_sample_buffer : register(t1,space1);

// ------------------------------------------------------------------------------------------------
// defined in the local root signature
// ------------------------------------------------------------------------------------------------
StructuredBuffer<Vertex> vertices : register(t1);
StructuredBuffer<uint> indices: register(t2);

cbuffer Geometry_constants : register(b2)
{
    uint geometry_index_offset;
}

cbuffer Material_constants : register(b3)
{
    // shared for all material compiled from the same MDL material
    int scattering_function_index;
    int opacity_function_index;
    int emission_function_index;
    int emission_intensity_function_index;
    int thin_walled_function_index;

    // individual properties of the different material instances
    int material_id;
    uint material_flags;
}

// ------------------------------------------------------------------------------------------------
// helper
// ------------------------------------------------------------------------------------------------

// selects one light source randomly
float3 sample_lights(
    Shading_state_material state, out float3 to_light, out float pdf, inout uint seed)
{
    float p_select_light = 1.0f;
    if (point_light_enabled != 0)
    {
        // keep it simple and use either point light or environment light, each with the same
        // probability. If the environment factor is zero, we always use the point light
        // Note: see also miss shader
        float p_select_light = environment_intensity_factor > 0.0f ? 0.5f : 1.0f;

        // in general, you would select the light depending on the importance of it
        // e.g. by incorporating their luminance

        // randomly select one of the lights
        if (rnd(seed) <= p_select_light)
        {
            pdf = DIRAC; // infinity

            // compute light direction and distance
            to_light = point_light_position - state.position;
            const float inv_distance2 = 1.0f / dot(to_light, to_light);
            to_light *= sqrt(inv_distance2);

            return point_light_intensity * inv_distance2 * 0.25f * M_ONE_OVER_PI / p_select_light;
        }

        // probability to select the environment instead
        p_select_light = (1.0f - p_select_light);
    }

    // light from the environment
    float3 radiance = environment_sample(   // (see common.hlsl)
        environment_texture,                // assuming lat long map
        environment_sample_buffer,          // importance sampling data of the environment map
        seed,
        to_light,
        pdf);

    pdf *= p_select_light;
    return radiance / pdf; // constant color
}


bool is_back_face()
{
    // get first index of the triangle, vertex positions, geometry normal in object space
    const uint index_offset = 3 * PrimitiveIndex() + geometry_index_offset;

    const uint3 vertex_indices = uint3(indices[index_offset + 0],
                                 indices[index_offset + 1],
                                 indices[index_offset + 2]);

    const float3 pos0 = vertices[vertex_indices.x].position;
    const float3 pos1 = vertices[vertex_indices.y].position;
    const float3 pos2 = vertices[vertex_indices.z].position;
    const float3 geom_normal = normalize(cross(pos1 - pos0, pos2 - pos0));

    return dot(geom_normal, ObjectRayDirection()) > 0.0f;
}

void setup_mdl_shading_state(
    inout Shading_state_material mdl_state, 
    Attributes attrib,
    out float3 shading_normal)
{
    const float3 barycentric = float3(1.0f - attrib.bary.x - attrib.bary.y, 
                                      attrib.bary.x, 
                                      attrib.bary.y);

    // first index of the triangle
    const uint index_offset = 3 * PrimitiveIndex() + geometry_index_offset;

    const float4x4 object_to_world = float4x4(ObjectToWorld(), 0.0f, 0.0f, 0.0f, 1.0f);
    const float4x4 world_to_object = float4x4(WorldToObject(), 0.0f, 0.0f, 0.0f, 1.0f);

    const uint3 vertex_indices = uint3(indices[index_offset + 0],
                                 indices[index_offset + 1],
                                 indices[index_offset + 2]);

    const float3 pos0 = vertices[vertex_indices.x].position;
    const float3 pos1 = vertices[vertex_indices.y].position;
    const float3 pos2 = vertices[vertex_indices.z].position;
    const float3 geom_normal = normalize(cross(pos1 - pos0, pos2 - pos0));

    const float3 normal = normalize(vertices[vertex_indices.x].normal * barycentric.x +
                              vertices[vertex_indices.y].normal * barycentric.y +
                              vertices[vertex_indices.z].normal * barycentric.z);

    // transform normals using inverse transpose
    float3 world_geom_normal = normalize(mul(float4(geom_normal, 0), world_to_object).xyz);
    const float3 world_normal = normalize(mul(float4(normal, 0), world_to_object).xyz);

    const float2 texcoord0 = vertices[vertex_indices.x].texcoord0 * barycentric.x +
        vertices[vertex_indices.y].texcoord0 * barycentric.y +
        vertices[vertex_indices.z].texcoord0 * barycentric.z;

    // flip geometry normal to the side of the incident ray
    if (dot(world_geom_normal, WorldRayDirection()) > 0.0)
        world_geom_normal *= -1.0f;

    // reconstruct tangent frame from vertex data
    float3 world_tangent, world_binormal;
    float4 tangent0 = vertices[vertex_indices.x].tangent0 * barycentric.x +
        vertices[vertex_indices.y].tangent0 * barycentric.y +
        vertices[vertex_indices.z].tangent0 * barycentric.z;
    tangent0.xyz = normalize(tangent0.xyz);
    world_tangent = normalize(mul(object_to_world, float4(tangent0.xyz, 0)).xyz);
    world_tangent = normalize(world_tangent - dot(world_tangent, world_normal) * world_normal);
    world_binormal = cross(world_normal, world_tangent) * tangent0.w;

    // fill the actual state fields
    mdl_state.normal = world_normal;
    mdl_state.geom_normal = world_geom_normal;
    mdl_state.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    mdl_state.animation_time = 0.0f;
    mdl_state.text_coords[0] = float3(texcoord0, 0);
    mdl_state.tangent_u[0] = world_tangent;
    mdl_state.tangent_v[0] = world_binormal;
    mdl_state.text_results = (float4[MDL_NUM_TEXTURE_RESULTS]) 0;
    mdl_state.ro_data_segment_offset = 0;
    mdl_state.world_to_object = world_to_object;
    mdl_state.object_to_world = object_to_world;
    mdl_state.object_id = 0;
    mdl_state.arg_block_offset = 0;

    // pass out the shading normal, this has to be reset before calling a second df::init
    shading_normal = world_normal;
}


// ------------------------------------------------------------------------------------------------
// MDL hit group shader
// ------------------------------------------------------------------------------------------------


[shader("anyhit")]
void MdlRadianceAnyHitProgram(inout RadianceHitInfo payload, Attributes attrib)
{
    // back face culling
    if (has_flag(material_flags, MATERIAL_FLAG_SINGLE_SIDED) && is_back_face())
    {
        IgnoreHit();
        return;
    }

    // early out if there is no opacity function
    if (has_flag(material_flags, MATERIAL_FLAG_OPAQUE))
        return;

    // setup MDL state
    Shading_state_material mdl_state;
    float3 shading_normal;
    setup_mdl_shading_state(mdl_state, attrib, shading_normal);


    // evaluate the cutout opacity
    const float opacity = mdl_geometry_cutout_opacity(opacity_function_index, mdl_state);

    // do alpha blending the stochastically way
    if (rnd(payload.seed) < opacity)
        return;

    IgnoreHit();
}

[shader("closesthit")] 
void MdlRadianceClosestHitProgram(inout RadianceHitInfo payload, Attributes attrib)
{
    // setup MDL state
    Shading_state_material mdl_state;
    float3 shading_normal;
    setup_mdl_shading_state(mdl_state, attrib, shading_normal);

    // add emission
    //---------------------------------------------------------------------------------------------
    if (emission_function_index >= 0 && emission_intensity_function_index >= 0)
    {
        // init for the use of the materials EDF
        mdl_edf_init(emission_function_index, mdl_state);

        // evaluate EDF
        Edf_evaluate_data eval_data = (Edf_evaluate_data) 0;
        eval_data.k1 = -WorldRayDirection();
        mdl_edf_evaluate(emission_function_index, eval_data, mdl_state);

        // evaluate intensity expression
        float3 intensity = mdl_emission_intensity(emission_intensity_function_index, mdl_state);

        // add emission
        payload.contribution += payload.weight * intensity * eval_data.edf;
    }

    // pre-compute and cache data that shared among 'mdl_bsdf_evaluate' and 'mdl_bsdf_sample' calls
    mdl_state.normal = shading_normal; // reset normal (init calls can change the normal due to maps)
    mdl_bsdf_init(scattering_function_index, mdl_state);

    // for thin walled materials there is no 'inside'
    const bool thin_walled = mdl_thin_walled(thin_walled_function_index, mdl_state);

    const bool inside = has_flag(payload.flags, FLAG_INSIDE);
    const float ior1 = (inside && !thin_walled) ? BSDF_USE_MATERIAL_IOR : 1.0f;
    const float ior2 = (inside && !thin_walled) ? 1.0f : BSDF_USE_MATERIAL_IOR;

    // Sample Light Sources
    //---------------------------------------------------------------------------------------------

    float3 to_light = float3(0.0f, 0.0f, 0.0f);
    float pdf = 0.0f;
    const float3 radiance_over_pdf = sample_lights(mdl_state, to_light, pdf, payload.seed);

    const float cos_theta = dot(to_light, mdl_state.geom_normal);
    if (((cos_theta > 0.0f) != inside) && pdf != 0.0f)
    {
        // cast a shadow ray; assuming light is always outside
        RayDesc ray;
        ray.Origin = offset_ray(mdl_state.position, mdl_state.geom_normal * (inside ? -1.0f : 1.0f));
        ray.Direction = to_light;
        ray.TMin = 0.0f;
        ray.TMax = 10000.0f;

        ShadowHitInfo shadow_payload;
        shadow_payload.isHit = false;
        shadow_payload.seed = payload.seed;

        TraceRay(
            SceneBVH,               // AccelerationStructure
            RAY_FLAG_NONE,          // RayFlags 
            0xFF /* allow all */,   // InstanceInclusionMask
            RAY_TYPE_SHADOW,        // RayContributionToHitGroupIndex
            RAY_TYPE_COUNT,         // MultiplierForGeometryContributionToHitGroupIndex
            RAY_TYPE_SHADOW,        // MissShaderIndex
            ray,
            shadow_payload);

        // not shadowed -> compute lighting 
        if (!shadow_payload.isHit)
        {
            // call generated mdl function to evaluate the scattering BSDF
            Bsdf_evaluate_data eval_data = (Bsdf_evaluate_data) 0;
            eval_data.ior1 = ior1;
            eval_data.ior2 = ior2;
            eval_data.k1 = -WorldRayDirection();
            eval_data.k2 = to_light;

            mdl_bsdf_evaluate(scattering_function_index, eval_data, mdl_state);

            // add to ray contribution
            const float mis_weight = pdf == DIRAC 
                ? 1.0f 
                : pdf / (pdf + eval_data.pdf);

            payload.contribution += 
                payload.weight * radiance_over_pdf * eval_data.bsdf * mis_weight;
        }
    }


    // Sample direction of the next ray
    //---------------------------------------------------------------------------------------------

    Bsdf_sample_data sample_data = (Bsdf_sample_data) 0;
    sample_data.ior1 = ior1;                    // IOR current medium
    sample_data.ior2 = ior2;                    // IOR other side
    sample_data.k1 = -WorldRayDirection();      // outgoing direction
    sample_data.xi = rnd3(payload.seed);        // random sample number

    mdl_bsdf_sample(scattering_function_index, sample_data, mdl_state);

    // stop on absorb
    if (sample_data.event_type == BSDF_EVENT_ABSORB)
    {
        add_flag(payload.flags, FLAG_DONE);
        return;
    }

    // flip inside/outside on transmission
    // setup next path segment
    payload.ray_direction_next = sample_data.k2;
    payload.weight *= sample_data.bsdf_over_pdf;
    if ((sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0)
    {
        toggle_flag(payload.flags, FLAG_INSIDE);
        // continue on the opposite side
        payload.ray_origin_next = offset_ray(mdl_state.position, -mdl_state.geom_normal);
    }
    else
    {
        // continue on the current side
        payload.ray_origin_next = offset_ray(mdl_state.position, mdl_state.geom_normal);
    }

    if ((sample_data.event_type & BSDF_EVENT_SPECULAR) != 0)
        payload.last_pdf = -1.0f;
    else
        payload.last_pdf = sample_data.pdf;
}

// ------------------------------------------------------------------------------------------------
// MDL shadow group shader
// ------------------------------------------------------------------------------------------------

[shader("anyhit")]
void MdlShadowAnyHitProgram(inout ShadowHitInfo payload, Attributes attrib)
{
    // back face culling
    if (has_flag(material_flags, MATERIAL_FLAG_SINGLE_SIDED) && is_back_face())
    {
        IgnoreHit();
        return;
    }

    // early out if there is no opacity function
    if (has_flag(material_flags, MATERIAL_FLAG_OPAQUE))
    {
        payload.isHit = true;
        AcceptHitAndEndSearch();
        return;
    }

    // setup MDL state
    Shading_state_material mdl_state;
    float3 shading_normal;
    setup_mdl_shading_state(mdl_state, attrib, shading_normal);

    // evaluate the cutout opacity
    const float opacity = mdl_geometry_cutout_opacity(opacity_function_index, mdl_state);

    // do alpha blending the stochastically way
    if (rnd(payload.seed) < opacity)
    {
        payload.isHit = true;
        AcceptHitAndEndSearch();
        return;
    }

    IgnoreHit();
}
