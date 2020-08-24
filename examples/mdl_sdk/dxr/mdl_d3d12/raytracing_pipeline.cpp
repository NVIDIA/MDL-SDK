/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "raytracing_pipeline.h"
#include "base_application.h"
#include "buffer.h"
#include "shader.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Raytracing_pipeline::Library::Library(
    const IDxcBlob* dxil_library,
    bool owns_dxil_library,
    const std::vector<std::string>& exported_symbols)

    : m_dxil_library(dxil_library)
    , m_owns_dxil_library(owns_dxil_library)
    , m_exported_symbols(exported_symbols.size())
    , m_exports(exported_symbols.size())
{
    // Create one export descriptor per symbol
    for (size_t i = 0; i < exported_symbols.size(); i++)
    {
        m_exported_symbols[i] = mi::examples::strings::str_to_wstr(exported_symbols[i]);
        m_exports[i] = {};
        m_exports[i].Name = m_exported_symbols[i].c_str();
        m_exports[i].ExportToRename = nullptr;
        m_exports[i].Flags = D3D12_EXPORT_FLAG_NONE;
    }

    // Create a library descriptor combining the DXIL code and the export names
    m_desc.DXILLibrary.BytecodeLength =
        const_cast<IDxcBlob*>(m_dxil_library)->GetBufferSize();

    m_desc.DXILLibrary.pShaderBytecode =
        const_cast<IDxcBlob*>(m_dxil_library)->GetBufferPointer();

    m_desc.NumExports = static_cast<UINT>(m_exported_symbols.size());
    m_desc.pExports = m_exports.data();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

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
    Root_signature* signature, bool owns_signature, const std::vector<std::string>& symbols)
    : m_root_signature(signature)
    , m_owns_root_signature(owns_signature)
    , m_signature(signature->get_signature())
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

Raytracing_pipeline::Raytracing_pipeline(Base_application* app, std::string debug_name)
    : m_app(app)
    , m_debug_name(debug_name)
    , m_is_finalized(false)
    , m_dummy_local_root_signature(nullptr)
    , m_global_root_signature(new Root_signature(app, debug_name + "_GlobalRootSignature"))
{
}

// ------------------------------------------------------------------------------------------------

Raytracing_pipeline::~Raytracing_pipeline()
{
    for (auto&& asso : m_signature_associations)
        if (asso.m_owns_root_signature)
            delete asso.m_root_signature;

    // might not be initialized yet because of failure
    if (m_dummy_local_root_signature != nullptr) delete m_dummy_local_root_signature;
    delete m_global_root_signature;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_pipeline::add_library(
    const IDxcBlob * dxil_library,
    bool owns_dxil_library,
    const std::vector<std::string>& exported_symbols)
{
    if (m_is_finalized) {
        log_error("Pipeline '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    if (!dxil_library) {
        log_error("Tried to add an invalid DxIL library "
                    " to pipeline:" + m_debug_name + ". Compiling failed?", SRC);
        return false;
    }

    // check that the library does not exist yet
    for (auto&& libs : m_libraries)
        if (libs.m_dxil_library == dxil_library) {
            log_error("Tried to add DxIL library multiple times "
                        "to pipeline:" + m_debug_name + ".", SRC);
            return false;
        }

    Library lib(dxil_library, owns_dxil_library, exported_symbols);

    // check if the symbols are not existing yet and add them to list of all exported symbols
    for (const auto& s : lib.m_exported_symbols)
    {
        if (m_all_exported_symbols.find(s) != m_all_exported_symbols.end())
        {
            log_error("Tried to add duplicated symbol '" + mi::examples::strings::wstr_to_str(s) +
                        "' to pipeline:" + m_debug_name + ".", SRC);
            // if (lib.m_owns_dxil_library)
            //     delete lib.m_dxil_library;
            return false;
        }
        m_all_exported_symbols.insert(s);
    }
    m_libraries.emplace_back(std::move(lib));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_pipeline::add_hitgroup(
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

    Hitgroup group(name, closest_hit_symbol, any_hit_symbol, intersection_symbol);

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

bool Raytracing_pipeline::add_signature_association(
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

    Root_signature_association asso(signature, owns_signature, symbols);

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

bool Raytracing_pipeline::finalize()
{
    if (m_is_finalized)
    {
        log_warning("Pipeline '" + m_debug_name + "' is already finalized. "
                    "Finalizing again is a NO-OP.", SRC);
        return true;
    }

    // The pipeline is made of a set of sub-objects, representing the DXIL libraries, hit group
    // declarations, root signature associations, plus some configuration objects
    UINT64 subobject_count =
        m_libraries.size() +                    // DXIL libraries
        m_hitgroups.size() +                    // Hit group declarations
        1 +                                     // Shader configuration
        1 +                                     // Shader payload
        2 * m_signature_associations.size() +   // Root signature declaration + association
        2 +                                     // Empty global and local root signatures
        1;                                      // Final pipeline sub-object

    std::vector<D3D12_STATE_SUBOBJECT> subobjects;
    subobjects.reserve(subobject_count);


    // Add all the DXIL libraries
    for (const auto& lib : m_libraries)
    {
        D3D12_STATE_SUBOBJECT libSubobject = {};
        libSubobject.Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
        libSubobject.pDesc = &lib.m_desc;
        subobjects.push_back(std::move(libSubobject));
    }

    // Add all the hit group declarations
    for (const auto& group : m_hitgroups)
    {
        D3D12_STATE_SUBOBJECT hitGroup = {};
        hitGroup.Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
        hitGroup.pDesc = &group.m_desc;
        subobjects.push_back(std::move(hitGroup));

        // Add hit group as exported symbol
        m_all_exported_symbols.insert(group.m_name);
    }

    // Add a sub-object for the shader payload configuration
    D3D12_RAYTRACING_SHADER_CONFIG shader_config_desc = {};
    shader_config_desc.MaxPayloadSizeInBytes = static_cast<UINT>(m_max_payload_size_in_byte);
    shader_config_desc.MaxAttributeSizeInBytes =
        static_cast<UINT>(m_max_attribute_size_in_byte);

    D3D12_STATE_SUBOBJECT shader_config = {};
    shader_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    shader_config.pDesc = &shader_config_desc;
    subobjects.push_back(std::move(shader_config));

    // Build a list of all the symbols for ray generation, miss and hit groups
    // All those shader have to be associated with the payload definition
    std::vector<LPCWSTR> exported_symbols;
    for (const auto& name : m_all_exported_symbols)
    {
        if (m_all_associated_symbols.find(name) == m_all_associated_symbols.end()) {
            log_error("Symbol or hit group '" + mi::examples::strings::wstr_to_str(name) +
                        "' is missing a root signature association "
                        "in pipeline:" + m_debug_name + ".", SRC);
            return false;
        }
        exported_symbols.push_back(name.c_str());
    }

    // Add a sub-object for the association between shader and the payload
    D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION shader_payload_association_desc = {};
    shader_payload_association_desc.NumExports = static_cast<UINT>(exported_symbols.size());
    const WCHAR** exported_symbols_data = exported_symbols.data();
    shader_payload_association_desc.pExports = exported_symbols_data;

    // Associate the set of shader with the payload defined in the previous sub-object
    shader_payload_association_desc.pSubobjectToAssociate = &subobjects[subobjects.size()-1];

    // Create and store the payload association object
    D3D12_STATE_SUBOBJECT shader_payload_association = {};
    shader_payload_association.Type =
        D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
    shader_payload_association.pDesc = &shader_payload_association_desc;
    subobjects.push_back(std::move(shader_payload_association));


    // The root signature association requires two objects for each: one to declare the root
    // signature, and another to associate that root signature to a set of symbols
    for (auto& asso : m_signature_associations)
    {
        // Add a sub-object to declare the root signature
        D3D12_STATE_SUBOBJECT root_signature = {};
        root_signature.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
        root_signature.pDesc = &asso.m_signature;
        subobjects.push_back(std::move(root_signature));

        // Add a sub-object for the association between the exported shader symbols and the root
        // signature
        asso.m_desc.NumExports = static_cast<UINT>(asso.m_symbol_pointers.size());
        asso.m_desc.pExports = asso.m_symbol_pointers.data();
        asso.m_desc.pSubobjectToAssociate = &subobjects[(subobjects.size() - 1)];

        D3D12_STATE_SUBOBJECT root_signature_association = {};
        root_signature_association.Type =
            D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION;
        root_signature_association.pDesc = &asso.m_desc;
        subobjects.push_back(std::move(root_signature_association));
    }

    // Add global root signature
    if (!m_global_root_signature->finalize()) return false;
    D3D12_STATE_SUBOBJECT global_root_signature_desc;
    global_root_signature_desc.Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    ID3D12RootSignature* global_signature = m_global_root_signature->get_signature();
    global_root_signature_desc.pDesc = &global_signature;
    subobjects.push_back(std::move(global_root_signature_desc));

    // The pipeline construction always requires an empty local root signature
    m_dummy_local_root_signature = new Root_signature(
        m_app, m_debug_name + "_dummy_local_root_signature");
    m_dummy_local_root_signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
    if (!m_dummy_local_root_signature->finalize()) return false;

    D3D12_STATE_SUBOBJECT local_root_signature_desc;
    local_root_signature_desc.Type = D3D12_STATE_SUBOBJECT_TYPE_LOCAL_ROOT_SIGNATURE;
    ID3D12RootSignature* local_signature = m_dummy_local_root_signature->get_signature();
    local_root_signature_desc.pDesc = &local_signature;
    subobjects.push_back(std::move(local_root_signature_desc));

    // Add a sub-object for the ray tracing pipeline configuration
    D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config_desc = {};
    pipeline_config_desc.MaxTraceRecursionDepth = static_cast<UINT>(m_max_recursion_depth);

    D3D12_STATE_SUBOBJECT pipeline_config = {};
    pipeline_config.Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    pipeline_config.pDesc = &pipeline_config_desc;
    subobjects.push_back(std::move(pipeline_config));

    // Describe the ray tracing pipeline state object
    D3D12_STATE_OBJECT_DESC pipeline_desc = {};
    pipeline_desc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    pipeline_desc.NumSubobjects = static_cast<UINT>(subobjects.size());
    pipeline_desc.pSubobjects = subobjects.data();

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
{
    if (ray_type_count == 0) {
        log_error("Ray type count can not be zero: " + m_debug_name, SRC);
    }
}

// ------------------------------------------------------------------------------------------------

Raytracing_acceleration_structure::~Raytracing_acceleration_structure()
{
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

    if (m_instance_buffer) {
        log_error("Acceleration structure already build. "
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

    m_bottom_level_structures[blas.m_index].m_geometry_descriptions.push_back(std::move(desc));
    return Raytracing_acceleration_structure::Geometry_handle(
        this,
        blas.m_index,
        m_bottom_level_structures[blas.m_index].m_geometry_descriptions.size() - 1);
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::build_bottom_level_structure(
    D3DCommandList* command_list,
    size_t blas_index)
{
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

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


    Bottom_level& blas = m_bottom_level_structures[blas_index];
    if (!blas.m_scratch_resource ||
        blas.m_scratch_resource->GetDesc().Width < prebuild_info.ScratchDataSizeInBytes)
    {
        if (!allocate_resource(
            &blas.m_scratch_resource,
            prebuild_info.ScratchDataSizeInBytes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            blas.m_debug_name_suffix + "_ScratchResource"))
            return false;
    }

    if (!allocate_resource(
        &blas.m_blas_resource,
        prebuild_info.ResultDataMaxSizeInBytes,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        blas.m_debug_name_suffix))
        return false;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {};
    build_desc.Inputs = accel_inputs;
    build_desc.ScratchAccelerationStructureData =
        blas.m_scratch_resource->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData =
        blas.m_blas_resource->GetGPUVirtualAddress();

    command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, 0);
    command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(
        blas.m_blas_resource.Get()));

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

    if (m_instance_buffer)
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

bool Raytracing_acceleration_structure::build_top_level_structure(
    D3DCommandList* command_list)
{
    if (m_instance_buffer) {
        log_error("Acceleration structure already build. Update not implemented: " +
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

    // upload instance data to GPU
    size_t buffer_size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * m_instances.size();
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(buffer_size);
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_instance_buffer)),
        "Failed to allocate instance data buffer for: " + m_debug_name, SRC))
        return false;

    set_debug_name(m_instance_buffer.Get(), m_debug_name + "_InstanceData");

    void *p_mapped_data;
    if (log_on_failure(m_instance_buffer->Map(0, nullptr, &p_mapped_data),
        "Failed to upload instance data for: " + m_debug_name, SRC))
        return false;

    memcpy(p_mapped_data, m_instances.data(), buffer_size);
    m_instance_buffer->Unmap(0, nullptr);

    // create the actual top level structure
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS accel_inputs = {};
    accel_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    accel_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    accel_inputs.NumDescs = static_cast<UINT>(m_instances.size());
    accel_inputs.Flags = buildFlags;
    accel_inputs.InstanceDescs = m_instance_buffer->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild_info = {};
    m_app->get_device()->GetRaytracingAccelerationStructurePrebuildInfo(
        &accel_inputs, &prebuild_info);

    if (!m_scratch_resource ||
        m_scratch_resource->GetDesc().Width < prebuild_info.ScratchDataSizeInBytes)
    {
        if (!allocate_resource(
            &m_scratch_resource,
            prebuild_info.ScratchDataSizeInBytes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            "_ScratchResource"))
            return false;
    }
    if (!allocate_resource(
        &m_top_level_structure,
        prebuild_info.ResultDataMaxSizeInBytes,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        "_TLAS"))
        return false;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build_desc = {};
    build_desc.Inputs = accel_inputs;
    build_desc.ScratchAccelerationStructureData = m_scratch_resource->GetGPUVirtualAddress();
    build_desc.DestAccelerationStructureData = m_top_level_structure->GetGPUVirtualAddress();

    command_list->BuildRaytracingAccelerationStructure(&build_desc, 0, 0);
    command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(
        m_top_level_structure.Get()));

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Raytracing_acceleration_structure::build(D3DCommandList* command_list)
{
    for (size_t i = 0, n = m_bottom_level_structures.size(); i < n; ++i)
        if (!build_bottom_level_structure(command_list, i)) return false;

    return build_top_level_structure(command_list);
}

// ------------------------------------------------------------------------------------------------

void Raytracing_acceleration_structure::release_static_scratch_buffers()
{
    if (!m_instance_buffer) {
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
    if (!m_instance_buffer) {
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
    if (!m_instance_buffer) {
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
    if (!m_instance_buffer) {
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
        initial_state,
        nullptr,
        IID_PPV_ARGS(resource)),
        "Failed to allocate memory for: " + m_debug_name + debug_name_suffix, SRC))
        return false;

    set_debug_name((*resource), m_debug_name + debug_name_suffix);
    return true;
}

}}} // mi::examples::mdl_d3d12
