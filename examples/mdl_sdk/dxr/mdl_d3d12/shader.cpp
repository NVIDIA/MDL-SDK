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

#include "shader.h"
#include "base_application.h"
#include "raytracing_pipeline.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <example_shared.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{

struct IncludeHandler : public IDxcIncludeHandler
{
public:
    explicit IncludeHandler(IDxcIncludeHandler* handler)
        : m_handler(handler)
        , m_cRef(0)
    {
    }

    // ---------------------------------------------------------------------------------------------

    HRESULT STDMETHODCALLTYPE LoadSource(
        _In_ LPCWSTR pFilename,
        _COM_Outptr_result_maybenull_ IDxcBlob **ppIncludeSource)
    {
        std::string filename = mi::examples::strings::wstr_to_str(pFilename);

        // drop strict relative marker
        if (mi::examples::strings::starts_with(filename, "./"))
            filename = filename.substr(2);

        // make path absolute if not already
        if (!mi::examples::io::is_absolute_path(filename))
            filename = mi::examples::io::get_executable_folder() + "/" + filename;

        std::wstring wFilename = mi::examples::strings::str_to_wstr(filename);
        return m_handler->LoadSource(wFilename.c_str(), ppIncludeSource);
    }

    // ---------------------------------------------------------------------------------------------

    HRESULT QueryInterface(REFIID riid, LPVOID * ppvObj)
    {
        if (!ppvObj) return E_INVALIDARG;
        *ppvObj = NULL;
        if (riid == IID_IUnknown || riid == __uuidof(IDxcIncludeHandler))
        {
            *ppvObj = (LPVOID)this;
            AddRef();
            return NOERROR;
        }
        return E_NOINTERFACE;
    }

    // ---------------------------------------------------------------------------------------------

    ULONG AddRef()
    {
        InterlockedIncrement(&m_cRef);
        return m_cRef;
    }

    // ---------------------------------------------------------------------------------------------

    ULONG Release()
    {
        ULONG ulRefCount = InterlockedDecrement(&m_cRef);
        if (0 == m_cRef)
            delete this;
        return ulRefCount;
    }

    // ---------------------------------------------------------------------------------------------

private:
    ComPtr<IDxcIncludeHandler> m_handler;
    LONG m_cRef;
};

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

IDxcBlob* Shader_compiler::compile_shader_library(
    const std::string& file_name,
    const std::map<std::string, std::string>* defines)
{
    std::string shader_source = "";
    {
        std::ifstream shader_file(file_name);
        if (shader_file.good())
        {
            std::stringstream strStream;
            strStream << shader_file.rdbuf();
            shader_source = strStream.str();
        }
    }
    if (shader_source.empty())
    {
        log_error("Failed to load shader source from file: " + file_name, SRC);
        return nullptr;
    }
    return compile_shader_library_from_string(shader_source, file_name, defines);
}

// ------------------------------------------------------------------------------------------------

IDxcBlob* Shader_compiler::compile_shader_library_from_string(
    const std::string& shader_source,
    const std::string& debug_name,
    const std::map<std::string, std::string>* defines)
{
    IDxcCompiler* compiler = nullptr;
    IDxcLibrary* library = nullptr;
    ComPtr<IncludeHandler> include_handler;

    if (log_on_failure(DxcCreateInstance(
        CLSID_DxcCompiler, __uuidof(IDxcCompiler), (void**)&compiler),
        "Failed to create IDxcCompiler", SRC))
        return nullptr;

    if (log_on_failure(DxcCreateInstance(
        CLSID_DxcLibrary, __uuidof(IDxcLibrary), (void**)&library),
        "Failed to create IDxcLibrary", SRC))
        return nullptr;

    IDxcIncludeHandler* base_handler;
    if (log_on_failure(library->CreateIncludeHandler(&base_handler),
        "Failed to create Include Handler.", SRC))
        return nullptr;

    include_handler = new IncludeHandler(base_handler);

    IDxcBlobEncoding* shader_source_blob;
    if (log_on_failure(library->CreateBlobWithEncodingFromPinned(
        (LPBYTE)shader_source.c_str(), (uint32_t)shader_source.size(), 0,
        &shader_source_blob),
        "Failed to create shader source blob: " + debug_name, SRC))
        return nullptr;

    // compilation arguments
#if DEBUG
    LPCWSTR pp_args[] = {
        L"/Zi", // debug info
    };
    UINT32 args_count = _countof(pp_args);

#else
    LPCWSTR* pp_args = nullptr;
    UINT32 args_count = 0;
#endif

    // since there are only a few defines, copying them seems okay
    std::vector<DxcDefine> wdefines;
    std::vector<std::wstring> wstrings;
    if (defines) {
        for (const auto d : *defines) {
            wstrings.push_back(mi::examples::strings::str_to_wstr(d.first));
            wstrings.push_back(mi::examples::strings::str_to_wstr(d.second));
            wdefines.push_back(DxcDefine{
                wstrings[wstrings.size() - 2].c_str(),
                wstrings[wstrings.size() - 1].c_str(),
                });
        }
    }

    IDxcOperationResult* result;
    std::wstring file_name_w = mi::examples::strings::str_to_wstr(debug_name);
    if (log_on_failure(compiler->Compile(
        shader_source_blob, file_name_w.c_str(), L"", L"lib_6_3",
        pp_args, args_count,
        wdefines.data(), wdefines.size(),
        include_handler.Get(), &result),

        "Failed to compile shader source: " + debug_name, SRC))
        return nullptr;

    HRESULT result_code = S_OK;
    if (log_on_failure(result->GetStatus(&result_code),
        "Failed to get compilation result for source: " + debug_name, SRC))
        return nullptr;

    if (FAILED(result_code))
    {
        IDxcBlobEncoding* error;
        if (log_on_failure(result->GetErrorBuffer(&error),
            "Failed to get compilation error for source: " + debug_name, SRC))
            return nullptr;

        std::vector<char> infoLog(error->GetBufferSize() + 1);
        memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
        infoLog[error->GetBufferSize()] = 0;
        std::string message = "Shader Compiler Error: " + debug_name + "\n";
        message.append(infoLog.data());
        log_error(message, SRC);
        return nullptr;
    }

    IDxcBlob* shader_blob;
    if (log_on_failure(result->GetResult(&shader_blob),
        "Failed to get shader blob for source: " + debug_name, SRC))
        return nullptr;

    static std::atomic<size_t> sl_counter = 0;

#if 0
    // setup debug name
    std::string file_name = mi::examples::io::normalize(debug_name);
    size_t p = file_name.find_last_of('/');
    if (p != std::string::npos)
        file_name = file_name.substr(p + 1);
    file_name = file_name.substr(0, file_name.find_last_of('.'));
    file_name += ".dxil";

    FILE* file = fopen(file_name.c_str(), "wb");
    if (file)
    {
        fwrite(shader_blob->GetBufferPointer(), shader_blob->GetBufferSize(), 1, file);
        fclose(file);
    }
#endif
    return shader_blob;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Shader::Shader(Base_application* app)
    : m_app(app)
{
}

// ------------------------------------------------------------------------------------------------

Shader::~Shader()
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Descriptor_table::Descriptor_table(const Descriptor_table& to_copy)
    : m_descriptor_ranges()
{
    size_t n = to_copy.m_descriptor_ranges.size();
    m_descriptor_ranges.resize(n);
    for (size_t i = 0; i < n; ++i)
        m_descriptor_ranges[i] = to_copy.m_descriptor_ranges[i];
}

// ------------------------------------------------------------------------------------------------

Descriptor_table::Descriptor_table(Descriptor_table&& to_move)
    : m_descriptor_ranges(std::move(to_move.m_descriptor_ranges))
{
}

// ------------------------------------------------------------------------------------------------

void Descriptor_table::register_cbv(
    size_t slot, size_t space, size_t heap_offset, size_t count)
{
    m_descriptor_ranges.push_back(CD3DX12_DESCRIPTOR_RANGE1{});
    m_descriptor_ranges.back().Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_CBV, static_cast<UINT>(count),
        static_cast<UINT>(slot), static_cast<UINT>(space),
        D3D12_DESCRIPTOR_RANGE_FLAG_NONE, static_cast<UINT>(heap_offset));
}

// ------------------------------------------------------------------------------------------------

void Descriptor_table::register_srv(
    size_t slot, size_t space, size_t heap_offset, size_t count)
{
    m_descriptor_ranges.push_back(CD3DX12_DESCRIPTOR_RANGE1{});
    m_descriptor_ranges.back().Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, static_cast<UINT>(count),
        static_cast<UINT>(slot), static_cast<UINT>(space),
        D3D12_DESCRIPTOR_RANGE_FLAG_NONE, static_cast<UINT>(heap_offset));
}

// ------------------------------------------------------------------------------------------------

void Descriptor_table::register_uav(
    size_t slot, size_t space, size_t heap_offset, size_t count)
{
    m_descriptor_ranges.push_back(CD3DX12_DESCRIPTOR_RANGE1{});
    m_descriptor_ranges.back().Init(
        D3D12_DESCRIPTOR_RANGE_TYPE_UAV, static_cast<UINT>(count),
        static_cast<UINT>(slot), static_cast<UINT>(space),
        D3D12_DESCRIPTOR_RANGE_FLAG_NONE, static_cast<UINT>(heap_offset));
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Root_signature::Element::Element()
    : kind(Kind::None)
    , size_in_word(0)
    , root_signature_index(0)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Root_signature::Root_signature(Base_application* app, const std::string& debug_name)
    : m_app(app)
    , m_debug_name(debug_name)
    , m_is_finalized(false)
    , m_flags(D3D12_ROOT_SIGNATURE_FLAG_NONE)
    , m_root_signature(nullptr)
{
}

// ------------------------------------------------------------------------------------------------

Root_signature::~Root_signature()
{
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_constants(size_t slot, size_t size_in_byte)
{
    if (m_is_finalized) {
        log_error("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    if (m_root_elements_b.find(slot) != m_root_elements_b.end()) {
        log_error("Root signature '" + m_debug_name +
                    "' already contains a constant at slot " + std::to_string(slot) + ".", SRC);
        return false;
    }

    Element e;
    e.kind = Element::Kind::Constant;
    e.size_in_word = size_in_byte / 4 + (size_in_byte % 4 == 0 ? 0 : 1);
    e.root_signature_index = m_root_parameters.size();
    m_root_elements_b[slot] = e;
    m_root_parameters.push_back(CD3DX12_ROOT_PARAMETER1());
    m_root_parameters.back().InitAsConstants(
        static_cast<UINT>(e.size_in_word), static_cast<UINT>(slot));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_cbv(size_t slot)
{
    if (m_is_finalized)
    {
        log_error("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    if (m_root_elements_b.find(slot) != m_root_elements_b.end())
    {
        log_error("Root signature '" + m_debug_name +
                    "' already contains a UAV at slot " + std::to_string(slot) + ".", SRC);
        return false;
    }

    Element e;
    e.kind = Element::Kind::CBV;
    e.size_in_word = 2;
    e.root_signature_index = m_root_parameters.size();
    m_root_elements_b[slot] = e;
    m_root_parameters.push_back(CD3DX12_ROOT_PARAMETER1());
    m_root_parameters.back().InitAsConstantBufferView(static_cast<UINT>(slot));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_uav(size_t slot)
{
    if (m_is_finalized) {
        log_error("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    if (m_root_elements_u.find(slot) != m_root_elements_u.end()) {
        log_error("Root signature '" + m_debug_name +
                    "' already contains a UAV at slot " + std::to_string(slot) + ".", SRC);
        return false;
    }

    Element e;
    e.root_signature_index = m_root_parameters.size();
    m_root_parameters.push_back(CD3DX12_ROOT_PARAMETER1());
    e.kind = Element::Kind::UAV;
    e.size_in_word = 2;
    m_root_parameters.back().InitAsUnorderedAccessView(static_cast<UINT>(slot));
    m_root_elements_u[slot] = e;

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_srv(size_t slot)
{
    if (m_is_finalized) {
        log_error("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    if (m_root_elements_t.find(slot) != m_root_elements_t.end()) {
        log_error("Root signature '" + m_debug_name +
                    "' already contains a SRC at slot " + std::to_string(slot) + ".", SRC);
        return false;
    }

    Element e;
    e.kind = Element::Kind::SRV;
    e.root_signature_index = m_root_parameters.size();
    e.size_in_word = 2;

    m_root_parameters.push_back(CD3DX12_ROOT_PARAMETER1());
    m_root_parameters.back().InitAsShaderResourceView(static_cast<UINT>(slot));
    m_root_elements_t[slot] = e;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_dt(const Descriptor_table& descriptor_table)
{
    if (m_is_finalized) {
        log_error("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return false;
    }

    // TODO check the table for duplicated assignments

    Element e;
    e.kind = Element::Kind::DescriptorTable;
    e.size_in_word = 1;
    e.root_signature_index = m_root_parameters.size();

    m_root_parameters.push_back(CD3DX12_ROOT_PARAMETER1());
    m_root_parameters.back().InitAsDescriptorTable(
        static_cast<UINT>(descriptor_table.m_descriptor_ranges.size()),
        descriptor_table.m_descriptor_ranges.data());
    m_root_elements_dt.emplace_back(std::move(e));
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::register_static_sampler(const D3D12_STATIC_SAMPLER_DESC& sampler_desc)
{
    m_static_samplers.push_back(sampler_desc);
    return true;
}

// ------------------------------------------------------------------------------------------------

D3D12_ROOT_SIGNATURE_FLAGS Root_signature::add_flag(D3D12_ROOT_SIGNATURE_FLAGS flag)
{
    if (m_is_finalized) {
        log_warning("Root signature '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return m_flags;
    }

    m_flags |= flag;
    return m_flags;
}

// ------------------------------------------------------------------------------------------------

bool Root_signature::finalize()
{
    if (m_is_finalized) {
        log_warning("Root signature '" + m_debug_name +
                    "' is already finalized. Finalizing again is a NO-OP.", SRC);
        return true;
    }

    // compute size (limited)
    // init descriptor tables
    size_t number_of_words = 0;

    for (auto&& e : m_root_elements_b)
        number_of_words += e.second.size_in_word;

    for (auto&& e : m_root_elements_t)
        number_of_words += e.second.size_in_word;

    for (auto&& e : m_root_elements_u)
        number_of_words += e.second.size_in_word;

    for (auto&& e : m_root_elements_dt)
        number_of_words += e.size_in_word;

    if (number_of_words > 64) {
        log_error("Root signature '" + m_debug_name + "' is too large. 64 words are allowed. "
                    + std::to_string(number_of_words) + " words are registered.", SRC);
        return false;
    }

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1(
        static_cast<UINT>(m_root_parameters.size()),
        (m_root_parameters.size() == 0 ? nullptr : m_root_parameters.data()),
        static_cast<UINT>(m_static_samplers.size()),
        (m_static_samplers.size() == 0 ? nullptr : m_static_samplers.data()),
        m_flags);

    // Serialize the root signature.
    ComPtr<ID3DBlob> serialized_root_signature;
    ComPtr<ID3DBlob> error;

    if (log_on_failure(D3DX12SerializeVersionedRootSignature(
        &desc, D3D_ROOT_SIGNATURE_VERSION_1_1, &serialized_root_signature, &error),
        "Failed to serialize root signature: " + m_debug_name, SRC))
    {
        std::vector<char> infoLog(error->GetBufferSize() + 1);
        memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
        infoLog[error->GetBufferSize()] = 0;
        std::string message = "Serialization error for root signature: " + m_debug_name + "\n";
        message.append(infoLog.data());
        log_error(message, SRC);
        return false;
    }

    // Create the root signature.
    if (log_on_failure(m_app->get_device()->CreateRootSignature(
        0, serialized_root_signature->GetBufferPointer(),
        serialized_root_signature->GetBufferSize(), IID_PPV_ARGS(&m_root_signature)),
        "Failed to create root signature: " + m_debug_name, SRC))
        return false;

    set_debug_name(m_root_signature.Get(), m_debug_name.c_str());
    m_is_finalized = true;
    return true;
}

// ------------------------------------------------------------------------------------------------

ID3D12RootSignature* Root_signature::get_signature()
{
    if (!m_is_finalized)
    {
        log_warning("Root signature '" + m_debug_name +
                    "' is not finalized. The signature not yet available.", SRC);
        return nullptr;
    }
    return m_root_signature.Get();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Shader_binding_tables::Shader_record::Shader_record()
    : m_mapped_table_pointer(nullptr)
    , m_local_root_arguments(0)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

const Shader_binding_tables::Shader_handle Shader_binding_tables::Shader_handle::invalid(
    nullptr, Kind::invalid, nullptr);

Shader_binding_tables::Shader_handle::Shader_handle(
    Shader_binding_tables* binding_table,
    Kind kind,
    void* shader_id)

    : m_kind(kind)
    , m_shader_binding_table(binding_table)
    , m_shader_id(shader_id)
{
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Shader_binding_tables::Shader_binding_tables(
    Raytracing_pipeline* pipeline,
    size_t ray_type_count,
    size_t hit_record_count,
    const std::string& debug_name)

    : m_app(pipeline->m_app)
    , m_is_finalized(false)
    , m_debug_name(debug_name)
    , m_pipeline(pipeline)
    , m_ray_type_count(ray_type_count)
    , m_hit_record_count(hit_record_count)
    , m_prefilled_dispatch_description{}
{
    // ray-gen and miss are empty by default the hit record count is defined by the number
    // of instance, geometries in BLAS and the number of rays
    m_shader_records[static_cast<size_t>(
        Shader_binding_tables::Shader_handle::Kind::miss)] =
            std::vector<Shader_binding_tables::Shader_record>(ray_type_count);

    m_shader_records[static_cast<size_t>(
        Shader_binding_tables::Shader_handle::Kind::hit_group)] =
            std::vector<Shader_binding_tables::Shader_record>(hit_record_count);
}

// ------------------------------------------------------------------------------------------------

Shader_binding_tables::~Shader_binding_tables()
{
}

// ------------------------------------------------------------------------------------------------

const Shader_binding_tables::Shader_handle Shader_binding_tables::add_ray_generation_program(
    const std::string& symbol_name)
{
    auto res = add_shader(Shader_handle::Kind::ray_generation, symbol_name);

    // add record entry
    if (res.is_valid())
    {
        auto& table = m_shader_records[static_cast<size_t>(
            Shader_binding_tables::Shader_handle::Kind::ray_generation)];
        table.push_back(Shader_record());
        table.back().m_shader_id = res.m_shader_id;
    }
    return res;
}

// ------------------------------------------------------------------------------------------------

const Shader_binding_tables::Shader_handle Shader_binding_tables::add_miss_program(
    size_t ray_type,
    const std::string& symbol_name)
{
    auto res = add_shader(Shader_handle::Kind::miss, symbol_name);

    // add record entries
    if (res.is_valid())
        m_shader_records[static_cast<size_t>(
            Shader_binding_tables::Shader_handle::Kind::miss)][ray_type].m_shader_id =
                res.m_shader_id;

    return res;
}

// ------------------------------------------------------------------------------------------------

const Shader_binding_tables::Shader_handle Shader_binding_tables::add_hit_group(
    size_t ray_type,
    const std::string& group_name)
{
    auto res = add_shader(Shader_handle::Kind::hit_group, group_name);

    // add record entries
    if (res.is_valid())
    {
        auto& table = m_shader_records[static_cast<size_t>(
            Shader_binding_tables::Shader_handle::Kind::hit_group)];
                for (size_t i = ray_type; i < table.size(); i += m_ray_type_count)
                    table[i].m_shader_id = res.m_shader_id;
    }
    return res;
}

// ------------------------------------------------------------------------------------------------

const Shader_binding_tables::Shader_handle Shader_binding_tables::add_shader(
    Shader_handle::Kind kind,
    const std::string& name)
{
    if (m_is_finalized) {
        log_error("Shader binding table '" + m_debug_name +
                    "' is already finalized. No further changes possible.", SRC);
        return Shader_binding_tables::Shader_handle::invalid;
    }

    // check if this is new
    if (m_added_symbol_names.find(name) != m_added_symbol_names.end()) {
        log_error("Identifier '" + name + "' was already added to: " + m_debug_name, SRC);
        return Shader_binding_tables::Shader_handle::invalid;
    }

    // Get the shader identifier, and check whether that identifier is known
    std::wstring name_w = mi::examples::strings::str_to_wstr(name);
    void* id = m_pipeline->m_pipeline_state_properties->GetShaderIdentifier(name_w.c_str());
    if (!id) {
        log_error("Identifier '" + name + "' is unknown to pipeline '" +
                    m_pipeline->m_debug_name + "' and therefore not allowed in: " +
                    m_debug_name, SRC);
        return Shader_binding_tables::Shader_handle::invalid;
    }

    assert(kind != Shader_handle::Kind::invalid);
    return Shader_binding_tables::Shader_handle(this, kind, id);
}

// ------------------------------------------------------------------------------------------------

bool Shader_binding_tables::set_shader_record(
    size_t index,
    const Shader_handle &shader_handle,
    const uint8_t* local_root_arguments,
    size_t size_in_byte)
{
    if (shader_handle.m_shader_binding_table != this ||
        shader_handle.get_kind() == Shader_handle::Kind::invalid)
    {
        log_error("Tried to set a binding table entry of a different table "
                    "or an invalid handle to: " + m_debug_name, SRC);
        return false;
    }

    assert(m_shader_records[static_cast<size_t>(shader_handle.get_kind())].size() > index);
    auto& record = m_shader_records[static_cast<size_t>(shader_handle.get_kind())][index];

    if (m_is_finalized)
    {
        memset(record.m_mapped_table_pointer, 0,
                m_prefilled_dispatch_description.MissShaderTable.StrideInBytes);
        memcpy(record.m_mapped_table_pointer, record.m_shader_id,
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
        memcpy(record.m_mapped_table_pointer + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                record.m_local_root_arguments.data(), record.m_local_root_arguments.size());
    }
    else
    {
        record.m_shader_id = shader_handle.m_shader_id;
        record.m_local_root_arguments.resize(size_in_byte);
        record.m_local_root_arguments.assign(
            local_root_arguments, local_root_arguments + size_in_byte);
    }

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Shader_binding_tables::finalize()
{
    if (m_is_finalized) {
        log_warning("Shader binding table '" + m_debug_name +
                    "' is already finalized. Finalizing again is a NO-OP.", SRC);
        return true;
    }

    // find the maximum number of parameters used by a single entry
    size_t max_local_root_argument_block_size = 0;
    for (const auto& k : m_shader_records)
        for (const auto& r : k)
            max_local_root_argument_block_size =
                std::max(max_local_root_argument_block_size, r.m_local_root_arguments.size());

    size_t shader_record_size_in_byte = round_to_power_of_two(
        max_local_root_argument_block_size + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
        D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

    size_t total_size_in_byte =
        (m_shader_records[0].size() + m_shader_records[1].size() + m_shader_records[2].size()) *
        shader_record_size_in_byte;

    total_size_in_byte = round_to_power_of_two(total_size_in_byte, 256);

    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(
        total_size_in_byte,
        D3D12_RESOURCE_FLAG_NONE,
        std::max(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT,
                    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT));

    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
        IID_PPV_ARGS(&m_binding_table_buffer)),
        "Failed to create buffer for shader binding table: " + m_debug_name, SRC))
        return false;
    set_debug_name(m_binding_table_buffer.Get(), m_debug_name);

    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_binding_table_buffer_upload)),
        "Failed to create upload buffer for shader binding table: " + m_debug_name, SRC))
        return false;
    set_debug_name(m_binding_table_buffer_upload.Get(), m_debug_name + "_upload");

    // map buffer and keep it mapped
    if (log_on_failure(m_binding_table_buffer_upload->Map(
        0, nullptr, reinterpret_cast<void**>(&m_mapped_binding_table)),
        "Failed to map buffer: " + m_debug_name, SRC))
        return false;

    uint8_t* p_table_start = m_mapped_binding_table;
    for (size_t k = 0; k < 3; ++k)
    {
        uint8_t* p_current = p_table_start;

        for (auto& entry : m_shader_records[k])
        {
            // keep the pointer for later updates
            entry.m_mapped_table_pointer = p_current;

            // clear to zero
            memset(p_current, 0, shader_record_size_in_byte);

            // copy the shader identifier
            memcpy(p_current, entry.m_shader_id, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);

            // copy the root arguments
            memcpy(p_current + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                    entry.m_local_root_arguments.data(), entry.m_local_root_arguments.size());

            p_current += shader_record_size_in_byte;
        }
        p_table_start += round_to_power_of_two(
            m_shader_records[k].size() * shader_record_size_in_byte,
            D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);
    }

    m_prefilled_dispatch_description.RayGenerationShaderRecord.StartAddress = 0;
    m_prefilled_dispatch_description.RayGenerationShaderRecord.SizeInBytes =
        m_shader_records[static_cast<size_t>(Shader_handle::Kind::ray_generation)].size() *
        shader_record_size_in_byte;

    m_prefilled_dispatch_description.MissShaderTable.StartAddress =
        m_prefilled_dispatch_description.RayGenerationShaderRecord.StartAddress +
        round_to_power_of_two(
            m_prefilled_dispatch_description.RayGenerationShaderRecord.SizeInBytes,
            D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

    m_prefilled_dispatch_description.MissShaderTable.SizeInBytes =
        m_shader_records[static_cast<size_t>(Shader_handle::Kind::miss)].size() *
        shader_record_size_in_byte;
    m_prefilled_dispatch_description.MissShaderTable.StrideInBytes = shader_record_size_in_byte;

    m_prefilled_dispatch_description.HitGroupTable.StartAddress =
        m_prefilled_dispatch_description.MissShaderTable.StartAddress +
        round_to_power_of_two(
            m_prefilled_dispatch_description.MissShaderTable.SizeInBytes,
            D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT);

    m_prefilled_dispatch_description.HitGroupTable.SizeInBytes =
        m_shader_records[static_cast<size_t>(Shader_handle::Kind::hit_group)].size() *
        shader_record_size_in_byte;
    m_prefilled_dispatch_description.HitGroupTable.StrideInBytes = shader_record_size_in_byte;


    m_prefilled_dispatch_description.RayGenerationShaderRecord.StartAddress +=
        m_binding_table_buffer->GetGPUVirtualAddress();

    m_prefilled_dispatch_description.MissShaderTable.StartAddress +=
        m_binding_table_buffer->GetGPUVirtualAddress();

    m_prefilled_dispatch_description.HitGroupTable.StartAddress +=
        m_binding_table_buffer->GetGPUVirtualAddress();

    m_prefilled_dispatch_description.Depth = 1;

    m_is_finalized = true;
    return true;
}

// ------------------------------------------------------------------------------------------------

void Shader_binding_tables::upload(D3DCommandList* command_list)
{
    if (!m_is_finalized) {
        log_warning("Shader binding table '" + m_debug_name +
                    "' is not finalized. Upload call ignored.", SRC);
        return;
    }

    command_list->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST));

    command_list->CopyResource(
        m_binding_table_buffer.Get(), m_binding_table_buffer_upload.Get());

    command_list->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
}

// ------------------------------------------------------------------------------------------------

D3D12_DISPATCH_RAYS_DESC Shader_binding_tables::get_dispatch_description() const
{
    return m_prefilled_dispatch_description;
}

}}} // mi::examples::mdl_d3d12
