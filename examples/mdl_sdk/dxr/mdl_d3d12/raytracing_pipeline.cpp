/******************************************************************************
 * Copyright (c) 2019-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#include "raytracing_pipeline.h"
#include "base_application.h"
#include "buffer.h"
#include "shader.h"
#include "descriptor_heap.h"
#include <numeric>
#include <execution>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{
    constexpr size_t tlas_instance_buffer_count = 4;

    bool acceleration_structure_flags_allow_update(
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags)
    {
        return (flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) != 0;
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS blas_build_flags(
        Raytracing_acceleration_structure::Build_policy policy)
    {
        if (policy == Raytracing_acceleration_structure::Build_policy::Fast_build_allow_update)
            return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

        return D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    }

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS tlas_build_flags(
        Raytracing_acceleration_structure::Build_policy policy)
    {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags =
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        if (policy ==  Raytracing_acceleration_structure::Build_policy::Fast_build_allow_update)
            flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        return flags;
    }

    const char* build_policy_to_string(
        Raytracing_acceleration_structure::Build_policy policy)
    {
        switch (policy)
        {
            case Raytracing_acceleration_structure::Build_policy::Fast_trace:
                return "fast-trace";
            case Raytracing_acceleration_structure::Build_policy::Fast_build_allow_update:
                return "fast-build-allow-update";
        }
        return "unknown";
    }
}

Raytracing_pipeline::Hitgroup::Hitgroup(
    std::string name,
    std::string closest_hit_symbol,
    std::string any_hit_symbol,
    std::string intersection_symbol)

    : m_name(mi::examples::strings::str_to_wstr(name))
    , m_closest_hit_symbol(mi::examples::strings::str_to_wstr(closest_hit_symbol))
    , m_any_hit_symbol(mi::examples::strings::str_to_wstr(any_hit_symbol))
    , m_intersection_symbol(mi::examples::strings::str_to_wstr(intersection_symbol))
{
    // Indicate which shader program is used for closest hit,
    // leave the other ones undefined (default behavior)
    m_desc.HitGroupExport = m_name.c_str();
    m_desc.ClosestHitShaderImport = m_closest_hit_symbol.empty()
        ? nullptr
        : m_closest_hit_symbol.c_str();

    m_desc.AnyHitShaderImport = m_any_hit_symbol.empty()
        ? nullptr
        : m_any_hit_symbol.c_str();

    m_desc.IntersectionShaderImport = m_intersection_symbol.empty()
        ? nullptr
        : m_intersection_symbol.c_str();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_pipeline::Root_signature_association::Root_signature_association(
    const Root_signature* signature, bool owns_signature, const std::vector<std::string>& symbols)
    : m_root_signature(signature)
    , m_signature(signature->get_signature())
    , m_owns_root_signature(owns_signature)
    , m_symbols(symbols.size())
    , m_symbol_pointers(symbols.size())
    , m_desc {}
{
    for (size_t i = 0; i < m_symbols.size(); i++)
    {
        m_symbols[i] = mi::examples::strings::str_to_wstr(symbols[i]);
        m_symbol_pointers[i] = m_symbols[i].c_str();
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Shader_collection::Shader_collection(Base_application* app, const Raytracing_pipeline* parent_pipeline, std::string debug_name)
    : m_app(app)
    , m_parent_pipeline(parent_pipeline)
    , m_debug_name(debug_name)
    , m_is_finalized(false)
{
}

// ------------------------------------------------------------------------------------------------

Shader_collection::~Shader_collection()
{
    for (auto&& asso : m_signature_associations)
        if (asso.m_owns_root_signature)
            delete asso.m_root_signature;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_pipeline::Raytracing_pipeline(Base_application* app, std::string debug_name)
    : m_app(app)
    , m_debug_name(debug_name)
    , m_is_finalized(false)
    , m_global_root_signature(new Root_signature(app, debug_name + "_GlobalRootSignature"))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_pipeline::~Raytracing_pipeline()
{
    if (m_global_root_signature)
        delete m_global_root_signature;

    for (size_t i = 0; i < m_collections.size(); i++)
        delete m_collections[i];
    m_collections.clear();
}

// ------------------------------------------------------------------------------------------------

Shader_collection& Raytracing_pipeline::create_collection(const std::string& debug_name)
{
    m_collections.push_back(new Shader_collection(m_app, this, debug_name));
    return *m_collections.back();
}
// ------------------------------------------------------------------------------------------------

bool Shader_collection::add_library(const Shader_library& dxil_library)
{
    if (m_is_finalized) {
        log_error("Pipeline '" + m_debug_name +
            "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    // check that the library does not exist yet
    for (auto&& libs : m_libraries)
        if (libs.get_dxil_library() == dxil_library.get_dxil_library()) {
            log_error("Tried to add DxIL library multiple times "
                "to pipeline:" + m_debug_name + ".", SRC);
            return false;
        }

    // add the library to the pipeline
    m_libraries.push_back(dxil_library);
    const Shader_library::Data* d3d_data = m_libraries.back().getData();

    // check if the symbols are not existing yet and add them to list of all exported symbols
    for (const auto& s : d3d_data->m_exported_symbols_w)
    {
        if (m_all_exported_symbols.find(s) != m_all_exported_symbols.end())
        {
            log_error("Tried to add duplicated symbol '" + mi::examples::strings::wstr_to_str(s) +
                "' to pipeline:" + m_debug_name + ".", SRC);
            return false;
        }
        m_all_exported_symbols.insert(s);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Shader_collection::add_hitgroup(
    std::string name,
    std::string closest_hit_symbol,
    std::string any_hit_symbol,
    std::string intersection_symbol)
{
    if (m_is_finalized)
    {
        log_error("Pipeline '" + m_debug_name +
            "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    // check that the group does not exist yet
    std::wstring wname = mi::examples::strings::str_to_wstr(name);
    for (auto&& group : m_hitgroups)
        if (group.m_name == wname)
        {
            log_error("Tried to add hit group '" + name + "' multiple times "
                "to pipeline:" + m_debug_name + ".", SRC);
            return false;
        }

    Raytracing_pipeline::Hitgroup group(name, closest_hit_symbol, any_hit_symbol, intersection_symbol);

    // check that the symbols do exist
    std::vector<const std::wstring*> group_symbols;
    if (!group.m_closest_hit_symbol.empty())
        group_symbols.push_back(&group.m_closest_hit_symbol);
    if (!group.m_any_hit_symbol.empty())
        group_symbols.push_back(&group.m_any_hit_symbol);
    if (!group.m_intersection_symbol.empty())
        group_symbols.push_back(&group.m_intersection_symbol);
    for (auto&& s : group_symbols)
        if (m_all_exported_symbols.find(*s) == m_all_exported_symbols.end())
        {
            log_error("Tried to add non existing symbol '" +
                mi::examples::strings::wstr_to_str(*s) +
                "' to hit-group '" + name +
                "' of pipeline:" + m_debug_name + ".", SRC);
            return false;
        }

    m_hitgroups.emplace_back(std::move(group));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Shader_collection::add_signature_association(
    Root_signature* signature,
    bool owns_signature,
    const std::vector<std::string>& symbols)
{
    if (m_is_finalized)
    {
        log_error("Pipeline '" + m_debug_name + "' is already finalized. "
            "No further changes possible.", SRC);
        return false;
    }

    Raytracing_pipeline::Root_signature_association asso(signature, owns_signature, symbols);

    // make sure the symbol (or hit-group) to associate is available
    for (const auto& s : asso.m_symbols)
    {
        if (m_all_exported_symbols.find(s) != m_all_exported_symbols.end())
            continue;

        bool found = false;
        for (size_t i = 0, n = m_hitgroups.size(); !found && i < n; ++i)
            for (const auto& group : m_hitgroups)
                if (group.m_name == s)
                    found = true;

        if (!found) {
            log_error("Tried to associate a symbol or hit group '" +
                mi::examples::strings::wstr_to_str(s) +
                " that is unknown to the pipeline:" + m_debug_name + ".", SRC);
            return false;
        }
    }

    // make the symbols associated
    for (const auto& s : asso.m_symbols)
        m_all_associated_symbols.insert(s);

    // keep signature map for shader binding table
    for (const auto& s : symbols)
        m_signature_map[s] = signature;

    m_signature_associations.emplace_back(std::move(asso));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Shader_collection::finalize(std::vector<D3D12_STATE_SUBOBJECT>* pipeline_subobjects)
{
    if (m_is_finalized)
    {
        log_warning("Shader Collection '" + m_debug_name + "' is already finalized. "
            "Finalizing again is a NO-OP.", SRC);
        return true;
    }

    const char* timing_name = pipeline_subobjects
        ? "compile shader collection (disabled)"
        : "compile shader collection";

    Timing t(timing_name);
    auto p = m_app->get_profiling().measure(timing_name);

    // compute the number of elements because they reference eachother by pointer
    // so we don't want resizing happening
    UINT64 collection_subobject_count =
        m_libraries.size() +                    // DXIL libraries
        m_hitgroups.size() +                    // hit group declarations
        2 * m_signature_associations.size() +   // root signature declaration + association
        1 +                                     // pipeline configuration
        1 +                                     // shader configuration
        1;                                      // global root signatures

    std::vector<D3D12_STATE_SUBOBJECT> collection_subobjects;
    collection_subobjects.reserve(collection_subobject_count);

    // using shader collections is way faster than compiling all libraries in one object
    // if a pipeline_subobject list is passed, we fall back to this slower mode
    std::vector<D3D12_STATE_SUBOBJECT>* subobjects = 
        pipeline_subobjects ? pipeline_subobjects : &collection_subobjects;

    // Add all the DXIL libraries
    for (const auto& lib : m_libraries)
    {
        D3D12_STATE_SUBOBJECT libSubobject = {};
        libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        libSubobject.pDesc = &lib.getData()->m_desc;
        subobjects->push_back(std::move(libSubobject));
    }

    // Add all the hit group declarations
    for (const auto& group : m_hitgroups)
    {
        D3D12_STATE_SUBOBJECT hitGroup = {};
        hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        hitGroup.pDesc = &group.m_desc;
        subobjects->push_back(std::move(hitGroup));

        // Add hit group as exported symbol
        m_all_exported_symbols.insert(group.m_name);
    }

    // The root signature association requires two objects for each: one to declare the root
    // signature, and another to associate that root signature to a set of symbols
    for (auto& asso : m_signature_associations)
    {
        // Add a sub-object to declare the root signature
        D3D12_STATE_SUBOBJECT root_signature = {};
        root_signature.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
        root_signature.pDesc = &asso.m_signature;
        subobjects->push_back(std::move(root_signature));

        // Add a sub-object for the association between the exported shader symbols and the root
        // signature
        asso.m_desc.NumExports = static_cast<UINT>(asso.m_symbol_pointers.size());
        asso.m_desc.pExports = asso.m_symbol_pointers.data();
        asso.m_desc.pSubobjectToAssociate = &subobjects->back();

        D3D12_STATE_SUBOBJECT root_signature_association = {};
        root_signature_association.Type =
            D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        root_signature_association.pDesc = &asso.m_desc;
        subobjects->push_back(std::move(root_signature_association));
    }

    // the other objects are only needed in shader collection mode
    if (pipeline_subobjects)
    {
        log_verbose("Elements of shader collection added to pipeline:" + m_debug_name);
        m_is_finalized = true;
        return true;
    }

    // Add a sub-object for the ray tracing pipeline configuration
    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config_desc = {};
    pipeline_config_desc.MaxTraceRecursionDepth = static_cast<UINT>(m_parent_pipeline->get_max_recursion_depth());
    D3D12_STATE_SUBOBJECT pipeline_config = {};
    pipeline_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    pipeline_config.pDesc = &pipeline_config_desc;
    subobjects->push_back(std::move(pipeline_config));

    // Add a sub-object for the shader payload configuration
    D3D12_RAYTRACING_SHADER_CONFIG shader_config_desc = {};
    shader_config_desc.MaxPayloadSizeInBytes = static_cast<UINT>(m_parent_pipeline->get_max_payload_size());
    shader_config_desc.MaxAttributeSizeInBytes = static_cast<UINT>(m_parent_pipeline->get_max_attribute_size());
    D3D12_STATE_SUBOBJECT shader_config = {};
    shader_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shader_config.pDesc = &shader_config_desc;
    subobjects->push_back(std::move(shader_config));

    // Add global root signature
    D3D12_STATE_SUBOBJECT global_root_signature_desc;
    global_root_signature_desc.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    const ID3D12RootSignature* global_signature = m_parent_pipeline->get_global_root_signature()->get_signature();
    global_root_signature_desc.pDesc = &global_signature;
    subobjects->push_back(std::move(global_root_signature_desc));

    // Create the shader collection object
    D3D12_STATE_OBJECT_DESC collection_desc = {};
    collection_desc.Type = D3D12_STATE_OBJECT_TYPE_COLLECTION;
    collection_desc.pSubobjects = collection_subobjects.data();
    if (collection_subobject_count != collection_subobjects.size())
    {
        log_error("Wrong number of subojects in shader collection: " + m_debug_name, SRC);
        return false;
    }
    collection_desc.NumSubobjects = static_cast<UINT>(collection_subobjects.size());
    if (log_on_failure( m_app->get_device()->CreateStateObject(
        &collection_desc, IID_PPV_ARGS(&m_state_object)),
        "Failed to create shader collection state object: " + m_debug_name, SRC))
        return false;
    set_debug_name(m_state_object.Get(), m_debug_name);
    m_is_finalized = true;

    log_verbose("Compiled shader collection successfully:" + m_debug_name);

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_pipeline::finalize()
{
    if (m_is_finalized)
    {
        log_warning("Pipeline '" + m_debug_name + "' is already finalized. "
                    "Finalizing again is a NO-OP.", SRC);
        return true;
    }

    // no changes to the root signature allowed
    if (!m_global_root_signature->finalize())
        return false;

    // The pipeline is made of a set of sub-objects, representing the DXIL libraries, hit group
    // declarations, root signature associations, plus some configuration objects
    UINT64 pipeline_subobject_count =
        1 + // pipeline configuration
        1;  // shader configuration

    // using shader collections is way faster than compiling all libraries in one object
    // for compile time performance profiling, it might be helpfull to test the slower mode as well
    bool no_shader_collections = m_app->get_options()->no_shader_collections;

    // depending on this setting suboject need to be added to collections, to the pipeline or to both
    if (no_shader_collections)
    {
        // we will add libraries, hitgroups, and associations to the pipeline directly
        for (const auto& c : m_collections)
        {
            pipeline_subobject_count += c->get_libraries().size();
            pipeline_subobject_count += c->get_hitgroups().size();
            pipeline_subobject_count += 2 * c->get_signature_associations().size();
        }
        // pipeline_subobject_count += 1;   // shader payload associciation
        pipeline_subobject_count += 1;   // global root signatures
    }
    else
    {
        pipeline_subobject_count += m_collections.size();

        // process all the shader collections in parallel
        // each collection will take care of its libraries, hitgroups, and associations
        std::vector<size_t> indexRange(m_collections.size());
        std::iota(indexRange.begin(), indexRange.end(), 0);
        std::for_each(std::execution::par, indexRange.begin(), indexRange.end(), [&](size_t index)
            {
                m_collections[index]->finalize();
            });
        // check that all collections are finalized successfully
        for (const auto& c : m_collections)
            if (!c->is_finialized())
                return false;
    }

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    subobjects.reserve(pipeline_subobject_count);

    // Add a sub-object for the ray tracing pipeline configuration
    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config_desc = {};
    pipeline_config_desc.MaxTraceRecursionDepth = static_cast<UINT>(get_max_recursion_depth());
    D3D12_STATE_SUBOBJECT pipeline_config = {};
    pipeline_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    pipeline_config.pDesc = &pipeline_config_desc;
    subobjects.push_back(std::move(pipeline_config));

    // Add a sub-object for the shader payload configuration
    D3D12_RAYTRACING_SHADER_CONFIG shader_config_desc = {};
    shader_config_desc.MaxPayloadSizeInBytes = static_cast<UINT>(get_max_payload_size());
    shader_config_desc.MaxAttributeSizeInBytes = static_cast<UINT>(get_max_attribute_size());
    D3D12_STATE_SUBOBJECT shader_config = {};
    shader_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shader_config.pDesc = &shader_config_desc;
    subobjects.push_back(std::move(shader_config));

    // IMPORTANT: collection_descs and global_signature must remain in scope until after 
    // CreateStateObject is called because subobjects contains pointers to these elements
    std::vector<D3D12_EXISTING_COLLECTION_DESC> collection_descs;   // only used in no_shader_collections = false
    const ID3D12RootSignature* global_signature = nullptr;          // only used in no_shader_collections = true

    if (no_shader_collections)
    {
        // add the libraries, hitgroups, and associations of each collection
        bool collections_added = true;
        for (const auto& c : m_collections)
            collections_added &= c->finalize(&subobjects);
        if (!collections_added)
            return false;

        // Add global root signature, which is otherwise added to the collections as well
        D3D12_STATE_SUBOBJECT global_root_signature_desc;
        global_root_signature_desc.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
        global_signature = get_global_root_signature()->get_signature();
        global_root_signature_desc.pDesc = &global_signature;
        subobjects.push_back(std::move(global_root_signature_desc));
    }
    else
    {
        // add all pre-compiled collections
        collection_descs.reserve(m_collections.size());
        for (const auto& c : m_collections)
        {
            collection_descs.push_back({});
            D3D12_EXISTING_COLLECTION_DESC& collection_desc = collection_descs.back();
            collection_desc.pExistingCollection = c->get_state();
            D3D12_STATE_SUBOBJECT libSubobject = {};
            libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_EXISTING_COLLECTION;
            libSubobject.pDesc = &collection_desc;
            subobjects.push_back(std::move(libSubobject));
        }
    }

    // Describe the ray tracing pipeline state object
    D3D12_STATE_OBJECT_DESC pipeline_desc = {};
    pipeline_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    pipeline_desc.NumSubobjects = static_cast<UINT>(subobjects.size());
    pipeline_desc.pSubobjects = subobjects.data();
    if (pipeline_subobject_count != subobjects.size())
    {
        log_error("Wrong number of subojects in pipeline: " + m_debug_name, SRC);
        return false;
    }

    // Create the state object
    if (log_on_failure(m_app->get_device()->CreateStateObject(
        &pipeline_desc, IID_PPV_ARGS(&m_pipeline_state)),
        "Failed to create raytracing pipeline state object: " + m_debug_name, SRC))
        return false;
    set_debug_name(m_pipeline_state.Get(), m_debug_name);

    // Cast the state object into a properties object,
    // allowing to later access the shader pointers by name
    if (log_on_failure(m_pipeline_state->QueryInterface(
        IID_PPV_ARGS(&m_pipeline_state_properties)),
        "Failed to get the raytracing state properties for: " + m_debug_name, SRC))
        return false;

    m_is_finalized = true;
    return true;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::BLAS_handle::BLAS_handle()
    : m_acceleration_structure(nullptr)
    , m_index(static_cast<size_t>(-1))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::BLAS_handle::BLAS_handle(
    Raytracing_acceleration_structure* acceleration_structure,
    size_t index)

    : m_acceleration_structure(acceleration_structure)
    , m_index(index)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Geometry_handle::Geometry_handle()
    : m_acceleration_structure(nullptr)
    , m_blas_index(static_cast<size_t>(-1))
    , m_geometry_index(static_cast<size_t>(-1))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Geometry_handle::Geometry_handle(
    Raytracing_acceleration_structure* acceleration_structure,
    size_t blas_index,
    size_t geometry_index)

    : m_acceleration_structure(acceleration_structure)
    , m_blas_index(blas_index)
    , m_geometry_index(geometry_index)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Instance_handle::Instance_handle()
    : m_acceleration_structure(nullptr)
    , m_blas_index(static_cast<size_t>(-1))
    , m_instance_index(static_cast<size_t>(-1))
    , instance_id(static_cast<size_t>(-1))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Instance_handle::Instance_handle(
    Raytracing_acceleration_structure* acceleration_structure,
    size_t blas_index,
    size_t instance_index,
    size_t instance_id)

    : m_acceleration_structure(acceleration_structure)
    , m_blas_index(blas_index)
    , m_instance_index(instance_index)
    , instance_id(instance_id)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Bottom_level::Bottom_level(std::string debug_name_suffix)
    : m_debug_name_suffix(debug_name_suffix)
    , m_build_flags(static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS>(0))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Bottom_level::~Bottom_level()
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::Raytracing_acceleration_structure(
    Base_application* app,
    size_t ray_type_count,
    std::string debug_name)

    : m_app(app)
    , m_debug_name(debug_name)
    , m_ray_type_count(ray_type_count)
    , m_geometry_contribution_multiplier_to_hit_record_index(ray_type_count)
    , m_instance_buffer_index(0)
    , m_build_policy(Build_policy::Fast_trace)
    , m_top_level_build_flags(static_cast<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS>(0))
    , m_top_level_rebuild_required(false)
{
    if (ray_type_count == 0) {
        log_error("Ray type count can not be zero: " + m_debug_name, SRC);
    }
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::~Raytracing_acceleration_structure()
{
    // free heap block
    m_app->get_resource_descriptor_heap()->free_views(m_top_level_structure_heap_index);
}

// ------------------------------------------------------------------------------------------------

const Raytracing_acceleration_structure::BLAS_handle
    Raytracing_acceleration_structure::add_bottom_level_structure(
        const std::string& debug_name_suffix)
{
    m_bottom_level_structures.emplace_back(Bottom_level(debug_name_suffix));
    return BLAS_handle(this, m_bottom_level_structures.size() - 1);
}
// ------------------------------------------------------------------------------------------------

const Raytracing_acceleration_structure::Geometry_handle
    Raytracing_acceleration_structure::add_geometry(
        const Raytracing_acceleration_structure::BLAS_handle& blas,
        Buffer* vertex_buffer,
        size_t vertex_buffer_offset_in_byte,
        size_t vertex_count,
        size_t vertex_stride_in_byte,
        size_t vertex_position_byte_offset,
        Index_buffer* index_buffer,
        size_t index_offset,
        size_t index_count)
{
    if (blas.m_acceleration_structure != this ||
        blas.m_index >= m_bottom_level_structures.size()) {
        log_error("Tried to add geometry to a foreign or invalid bottom level accelerator "
                    "structure to: " + m_debug_name, SRC);
        return Raytracing_acceleration_structure::Geometry_handle();
    }

    Bottom_level& bottom_level = m_bottom_level_structures[blas.m_index];
    if (bottom_level.m_blas_resource) {
        log_error("Bottom level acceleration structure already built. "
                    "Adding further geometries is not implemented: " + m_debug_name, SRC);
        return Raytracing_acceleration_structure::Geometry_handle();
    }

    D3D12_RAYTRACING_GEOMETRY_DESC desc = {};
    desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    desc.Triangles.IndexBuffer =
        index_buffer->get_resource()->GetGPUVirtualAddress() + // base address
        index_offset * sizeof(uint32_t); // offset to first index of the mesh (part)
    desc.Triangles.IndexCount = static_cast<UINT>(index_count);
    desc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    desc.Triangles.Transform3x4 = 0;
    desc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    desc.Triangles.VertexCount = static_cast<UINT>(vertex_count);
    desc.Triangles.VertexBuffer.StrideInBytes = vertex_stride_in_byte;
    desc.Triangles.VertexBuffer.StartAddress =
        vertex_buffer->get_resource()->GetGPUVirtualAddress() + // base address
        vertex_buffer_offset_in_byte + // first vertex of the mesh part
        vertex_position_byte_offset;

    desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;

    bottom_level.m_geometry_descriptions.push_back(std::move(desc));
    return Raytracing_acceleration_structure::Geometry_handle(
        this,
        blas.m_index,
        bottom_level.m_geometry_descriptions.size() - 1);
}

// ------------------------------------------------------------------------------------------------

void Raytracing_acceleration_structure::set_build_policy(Build_policy policy)
{
    m_build_policy = policy;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::build_bottom_level_structure(
    D3DCommandList* command_list,
    size_t blas_index,
    bool update)
{
    Bottom_level& blas = m_bottom_level_structures[blas_index];
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS desired_build_flags =
        blas_build_flags(m_build_policy);
    const bool perform_update =
        update &&
        blas.m_blas_resource &&
        blas.m_build_flags == desired_build_flags &&
        acceleration_structure_flags_allow_update(desired_build_flags);

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = desired_build_flags;
    if (perform_update)
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

    if (perform_update && !blas.m_blas_resource)
    {
        log_error("Tried to update BLAS before it was built: " +
            m_debug_name + blas.m_debug_name_suffix, SRC);
        return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS accel_inputs = {};
    accel_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    accel_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    accel_inputs.pGeometryDescs =
        m_bottom_level_structures[blas_index].m_geometry_descriptions.data();
    accel_inputs.NumDescs =
        static_cast<UINT>(m_bottom_level_structures[blas_index].m_geometry_descriptions.size());
    accel_inputs.Flags = buildFlags;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {};
    m_app->get_device()->GetRaytracingAccelerationStructurePrebuildInfo(
        &accel_inputs, &prebuild_info);


    const UINT64 scratch_size = (perform_update && prebuild_info.UpdateScratchDataSizeInBytes != 0)
        ? prebuild_info.UpdateScratchDataSizeInBytes
        : prebuild_info.ScratchDataSizeInBytes;

    if (!blas.m_scratch_resource ||
        blas.m_scratch_resource->GetDesc().Width < scratch_size)
    {
        if (!allocate_resource(
            command_list,
            &blas.m_scratch_resource,
            scratch_size,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            blas.m_debug_name_suffix + "_ScratchResource"))
            return false;
    }

    if (!blas.m_blas_resource ||
        (!perform_update && blas.m_blas_resource->GetDesc().Width <
            prebuild_info.ResultDataMaxSizeInBytes))
    {
        if (!allocate_resource(
            command_list,
            &blas.m_blas_resource,
            prebuild_info.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            blas.m_debug_name_suffix))
            return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {};
    build_desc.Inputs = accel_inputs;
    build_desc.ScratchAccelerationStructureData =
        blas.m_scratch_resource->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData =
        blas.m_blas_resource->GetGPUVirtualAddress();
    if (perform_update)
        build_desc.SourceAccelerationStructureData =
            blas.m_blas_resource->GetGPUVirtualAddress();

    auto resource_barrier = CD3DX12_RESOURCE_BARRIER::UAV(blas.m_blas_resource.Get());
    command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, 0);
    command_list->ResourceBarrier(1, &resource_barrier);

    blas.m_build_flags = desired_build_flags;
    return true;
}

// ------------------------------------------------------------------------------------------------

const Raytracing_acceleration_structure::Instance_handle
    Raytracing_acceleration_structure::add_instance(
        const BLAS_handle& blas,
        const DirectX::XMMATRIX& transform,
        UINT instance_mask,
        UINT flags,
        size_t instance_id)
{
    if (blas.m_acceleration_structure != this ||
        blas.m_index >= m_bottom_level_structures.size())
    {
        log_error("Tried to add an instance of a different or invalid "
                    "bottom level accelerator structure to: " + m_debug_name, SRC);
        return Raytracing_acceleration_structure::Instance_handle();
    }

    if (!m_instance_buffers.empty())
    {
        log_error("Acceleration structure already build. "
                    "Adding further instances is not implemented: " + m_debug_name, SRC);
        return Raytracing_acceleration_structure::Instance_handle();
    }

    D3D12_RAYTRACING_INSTANCE_DESC instance_desc = {};
    DirectX::XMMATRIX transform_T = DirectX::XMMatrixTranspose(transform);
    memcpy(instance_desc.Transform, &transform_T, sizeof(instance_desc.Transform));
    instance_desc.InstanceMask = instance_mask;
    instance_desc.Flags = flags;
    instance_desc.InstanceMask = 0xFF;
    instance_desc.InstanceID = instance_id;

    // these will be set before building the top level structure
    instance_desc.InstanceContributionToHitGroupIndex = 0;
    instance_desc.AccelerationStructure = 0;

    m_instances.push_back(std::move(instance_desc));
    m_instance_blas_indices.push_back(blas.m_index);
    m_instance_contribution_to_hit_record_index.push_back(0);

    return Raytracing_acceleration_structure::Instance_handle(
        this, blas.m_index, m_instances.size() - 1, instance_id);
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::set_instance_transform(
    const Instance_handle& instance_handle,
    const DirectX::XMMATRIX& transform)
{
    if (instance_handle.m_acceleration_structure != this ||
        instance_handle.m_instance_index >= m_instances.size())
    {
        log_error("Tried to modify an  different or invalid instance of: " +
                    m_debug_name, SRC);
        return false;
    }

    D3D12_RAYTRACING_INSTANCE_DESC& instance_desc =
        m_instances[instance_handle.m_instance_index];
    DirectX::XMMATRIX transform_T = DirectX::XMMatrixTranspose(transform);
    memcpy(instance_desc.Transform, &transform_T, sizeof(instance_desc.Transform));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::set_instance_bottom_level_structure(
    Instance_handle& instance_handle,
    const BLAS_handle& blas)
{
    if (instance_handle.m_acceleration_structure != this ||
        instance_handle.m_instance_index >= m_instances.size() ||
        instance_handle.m_blas_index >= m_bottom_level_structures.size() ||
        blas.m_acceleration_structure != this ||
        blas.m_index >= m_bottom_level_structures.size())
    {
        log_error("Tried to modify an invalid instance or assign an invalid bottom level "
                    "accelerator structure of: " + m_debug_name, SRC);
        return false;
    }

    const size_t previous_blas_index = instance_handle.m_blas_index;
    const size_t previous_geometry_count =
        m_bottom_level_structures[previous_blas_index].m_geometry_descriptions.size();
    const size_t next_geometry_count =
        m_bottom_level_structures[blas.m_index].m_geometry_descriptions.size();
    if (previous_geometry_count != next_geometry_count)
    {
        log_error("Tried to assign a bottom level acceleration structure with a different "
                    "geometry count to an existing instance of: " + m_debug_name, SRC);
        return false;
    }

    if (instance_handle.m_blas_index != blas.m_index)
    {
        m_instance_blas_indices[instance_handle.m_instance_index] = blas.m_index;
        instance_handle.m_blas_index = blas.m_index;
        m_top_level_rebuild_required = true;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::build_top_level_structure(
    D3DCommandList* command_list, bool update)
{
    if (!m_instance_buffers.empty() && !update) {
        log_error("Acceleration structure already build: " +
                    m_debug_name, SRC);
        return false;
    }

    if (m_instances.size() == 0) {
        log_error("Tried to build without any instance: " + m_debug_name, SRC);
        return false;
    }

    // compute the hit record offsets and update the instances
    size_t offset = 0;
    for (size_t i = 0, n = m_instances.size(); i < n; i++)
    {
        auto& blas = m_bottom_level_structures[m_instance_blas_indices[i]];

        m_instances[i].AccelerationStructure =
            blas.m_blas_resource->GetGPUVirtualAddress();
        m_instances[i].InstanceContributionToHitGroupIndex = offset;
        m_instance_contribution_to_hit_record_index[i] = offset;
        offset += m_geometry_contribution_multiplier_to_hit_record_index *
                    blas.m_geometry_descriptions.size();
    }

    // Upload instance data to GPU. Keep a small ring of upload buffers alive because the TLAS
    // build reads InstanceDescs asynchronously after this command list is submitted.
    size_t buffer_size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * m_instances.size();
    if (m_instance_buffers.empty())
        m_instance_buffers.resize(tlas_instance_buffer_count);

    ComPtr<ID3D12Resource>& instance_buffer = m_instance_buffers[m_instance_buffer_index];
    m_instance_buffer_index = (m_instance_buffer_index + 1) % m_instance_buffers.size();
    if (!instance_buffer || instance_buffer->GetDesc().Width < buffer_size)
    {
        auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(buffer_size);
        auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        if (log_on_failure(m_app->get_device()->CreateCommittedResource(
            &uploadHeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&instance_buffer)),
            "Failed to allocate instance data buffer for: " + m_debug_name, SRC))
            return false;

        set_debug_name(instance_buffer.Get(), m_debug_name + "_InstanceData");
    }

    void *p_mapped_data;
    if (log_on_failure(instance_buffer->Map(0, nullptr, &p_mapped_data),
        "Failed to upload instance data for: " + m_debug_name, SRC))
        return false;

    memcpy(p_mapped_data, m_instances.data(), buffer_size);
    instance_buffer->Unmap(0, nullptr);

    // create the actual top level structure
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS desired_build_flags =
        tlas_build_flags(m_build_policy);
    const bool perform_update =
        update &&
        !m_top_level_rebuild_required &&
        m_top_level_structure &&
        m_top_level_build_flags == desired_build_flags &&
        acceleration_structure_flags_allow_update(desired_build_flags);
    const bool build_flags_changed =
        update &&
        m_top_level_structure &&
        m_top_level_build_flags != desired_build_flags;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = desired_build_flags;
    if (perform_update)
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    if (build_flags_changed)
    {
        log_info("Rebuilding acceleration structure " + m_debug_name +
            " (policy=" + build_policy_to_string(m_build_policy) + ")");
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS accel_inputs = {};
    accel_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    accel_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    accel_inputs.NumDescs = static_cast<UINT>(m_instances.size());
    accel_inputs.Flags = buildFlags;
    accel_inputs.InstanceDescs = instance_buffer->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {};
    m_app->get_device()->GetRaytracingAccelerationStructurePrebuildInfo(
        &accel_inputs, &prebuild_info);

    const UINT64 scratch_size = (perform_update && prebuild_info.UpdateScratchDataSizeInBytes != 0)
        ? prebuild_info.UpdateScratchDataSizeInBytes
        : prebuild_info.ScratchDataSizeInBytes;

    if (!m_scratch_resource ||
        m_scratch_resource->GetDesc().Width < scratch_size)
    {
        if (!allocate_resource(
            command_list,
            &m_scratch_resource,
            scratch_size,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            "_ScratchResource"))
            return false;
    }
    const bool needs_tlas_allocation =
        !m_top_level_structure ||
        (!perform_update &&
            m_top_level_structure->GetDesc().Width < prebuild_info.ResultDataMaxSizeInBytes);
    if (needs_tlas_allocation)
    {
        if (!allocate_resource(
            command_list,
            &m_top_level_structure,
            prebuild_info.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            "_TLAS"))
            return false;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {};
    build_desc.Inputs = accel_inputs;
    build_desc.ScratchAccelerationStructureData = m_scratch_resource->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = m_top_level_structure->GetGPUVirtualAddress();

    const bool needs_tlas_srv_update =
        needs_tlas_allocation || !m_top_level_structure_heap_index.is_valid();

    if (!m_top_level_structure_heap_index.is_valid())
    {
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
        m_top_level_structure_heap_index = resource_heap.reserve_views(1);
        assert(m_top_level_structure_heap_index.is_valid());
    }

    if (needs_tlas_srv_update)
    {
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
        if (!resource_heap.create_shader_resource_view(this, m_top_level_structure_heap_index))
            return false;
    }

    if (perform_update)
    {
        // in place update
        build_desc.SourceAccelerationStructureData = m_top_level_structure->GetGPUVirtualAddress();
    }

    auto resource_barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_top_level_structure.Get());
    command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, 0);
    command_list->ResourceBarrier(1, &resource_barrier);

    m_top_level_build_flags = desired_build_flags;
    m_top_level_rebuild_required = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::build(D3DCommandList* command_list)
{
    log_info("Building acceleration structure " + m_debug_name +
        " (policy=" + build_policy_to_string(m_build_policy) + ")");

    for (size_t i = 0, n = m_bottom_level_structures.size(); i < n; ++i)
        if (!build_bottom_level_structure(command_list, i, false)) return false;

    return build_top_level_structure(command_list, false);
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::update(D3DCommandList* command_list)
{
    return build_top_level_structure(command_list, true);
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::update_bottom_level_structures(D3DCommandList* command_list)
{
    for (size_t i = 0, n = m_bottom_level_structures.size(); i < n; ++i)
        if (!build_bottom_level_structure(command_list, i, true)) return false;

    return build_top_level_structure(command_list, true);
}

// ------------------------------------------------------------------------------------------------

void Raytracing_acceleration_structure::release_static_scratch_buffers()
{
    if (m_instance_buffers.empty()) {
        log_warning("Acceleration structure is not yet build. "
                    "Call to release scratch buffers ignored: " + m_debug_name, SRC);
        return;
    }

    // dynamic updates are not implemented, so all scratch resources can be released
    for (auto&& blas : m_bottom_level_structures)
        if (blas.m_scratch_resource)
            blas.m_scratch_resource.Reset();

    if (m_scratch_resource)
        m_scratch_resource.Reset();
}

// ------------------------------------------------------------------------------------------------

size_t Raytracing_acceleration_structure::compute_hit_record_index(
    size_t ray_type,
    const Instance_handle& instance_handle,
    const Geometry_handle& geometry_handle)
{
    if (m_instance_buffers.empty()) {
        log_error("Acceleration structure is not yet build: " + m_debug_name, SRC);
        return false;
    }

    if (!instance_handle.is_valid() || instance_handle.m_acceleration_structure != this ||
        !geometry_handle.is_valid() || geometry_handle.m_acceleration_structure != this) {
        log_error("Provided handles are invalid or from a "
                    "different acceleration structure: " + m_debug_name, SRC);
        return false;
    }

    if (instance_handle.m_blas_index != geometry_handle.m_blas_index) {
        log_error("Instance handle and geometry handle point to a "
                    "different bottom level structure: " + m_debug_name, SRC);
        return false;
    }

    if (ray_type >= m_ray_type_count)
    {
        log_error("Provided ray type '" + std::to_string(ray_type) + "' has to be less than "
                    "the set ray type count '" + std::to_string(m_ray_type_count) + "': " +
                    m_debug_name, SRC);
        return false;
    }

    // has to match the value in the TraceRay call in the shader!
    assert(m_geometry_contribution_multiplier_to_hit_record_index == m_ray_type_count);

    return m_geometry_contribution_multiplier_to_hit_record_index *
            geometry_handle.m_geometry_index +
            m_instance_contribution_to_hit_record_index[instance_handle.m_instance_index] +
            ray_type;
}

// ------------------------------------------------------------------------------------------------

size_t Raytracing_acceleration_structure::get_hit_record_count() const
{
    if (m_instance_buffers.empty()) {
        log_error("Acceleration structure is not yet build: " + m_debug_name, SRC);
        return false;
    }

    size_t last_instance = m_instances.size() - 1;
    auto& last_blas = m_bottom_level_structures[m_instance_blas_indices[last_instance]];

    size_t hit_record_count = m_instance_contribution_to_hit_record_index[last_instance] +
        m_ray_type_count * last_blas.m_geometry_descriptions.size();

    return hit_record_count;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::get_shader_resource_view_description(
    D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const
{
    if (m_instance_buffers.empty()) {
        log_error("Acceleration structure is not yet build: " + m_debug_name, SRC);
        return false;
    }

    desc = {};
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.RaytracingAccelerationStructure.Location =
        m_top_level_structure->GetGPUVirtualAddress();
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::allocate_resource(
    D3DCommandList* command_list,
    ID3D12Resource** resource,
    UINT64 size_in_byte,
    D3D12_RESOURCE_STATES initial_state,
    const std::string& debug_name_suffix)
{
    auto upload_heap_properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(
        size_in_byte, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &upload_heap_properties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE
            ? D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE
            : D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(resource)),
        "Failed to allocate memory for: " + m_debug_name + debug_name_suffix, SRC))
        return false;

    set_debug_name((*resource), m_debug_name + debug_name_suffix);

    if (initial_state != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)
    {
        auto resource_barrier = CD3DX12_RESOURCE_BARRIER::Transition(*resource,
            D3D12_RESOURCE_STATE_COMMON, initial_state);
        command_list->ResourceBarrier(1, &resource_barrier);
    }

    return true;
}

}}} // mi::examples::mdl_d3d12
