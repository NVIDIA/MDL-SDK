/******************************************************************************
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <d3d12shader.h>

#ifdef MDL_ENABLE_SLANG
#include <slang.h>
#include <slang-com-ptr.h>
#endif

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

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------


#ifdef MDL_ENABLE_SLANG
class SlangSessionPool
{
public:
    SlangSessionPool(size_t initial_pool_size)
    {
        for (size_t i = 0; i < initial_pool_size; ++i)
            m_sessions.push_back(spCreateSession());
    }

    ~SlangSessionPool()
    {
        while (!m_sessions.empty())
        {
            spDestroySession(m_sessions.back());
            m_sessions.pop_back();
        }
    }

    SlangSession* acquire_session()
    {
        SlangSession* session = nullptr;
        if (!m_sessions.empty())
        {
            session = m_sessions.back();
            m_sessions.pop_back();
        }
        else
            session = spCreateSession();

        return session;
    }

    void release_session(SlangSession* session)
    {
        m_sessions.push_back(session);
    }

private:
    std::vector<SlangSession*> m_sessions;
};

SlangSessionPool g_slang_session_pool(std::thread::hardware_concurrency());

SlangOptimizationLevel shader_opt_to_slang_opt(const std::string& shader_opt)
{
    if (shader_opt == "Od" || shader_opt == "O0") return SLANG_OPTIMIZATION_LEVEL_NONE;
    else if (shader_opt == "O1") return SLANG_OPTIMIZATION_LEVEL_DEFAULT;
    else if (shader_opt == "O2") return SLANG_OPTIMIZATION_LEVEL_HIGH;
    else if (shader_opt == "O3") return SLANG_OPTIMIZATION_LEVEL_MAXIMAL;
    return SLANG_OPTIMIZATION_LEVEL_DEFAULT;
}
#endif //MDL_ENABLE_SLANG

} // anonymous


Shader_library::Shader_library(IDxcBlob* blob, const std::vector<std::string>& exported_symbols)
    : m_dxil_library(blob)
    , m_exported_symbols(exported_symbols)
{
    m_d3d_data = std::make_shared<Shader_library::Data>();
    m_d3d_data->m_exported_symbols_w.resize(exported_symbols.size());
    m_d3d_data->m_exports.resize(exported_symbols.size());

    // Create one export descriptor per symbol
    for (size_t i = 0; i < exported_symbols.size(); i++)
    {
        m_d3d_data->m_exported_symbols_w[i] =
            mi::examples::strings::str_to_wstr(exported_symbols[i]);
        m_d3d_data->m_exports[i] = {};
        m_d3d_data->m_exports[i].Name = m_d3d_data->m_exported_symbols_w[i].c_str();
        m_d3d_data->m_exports[i].ExportToRename = nullptr;
        m_d3d_data->m_exports[i].Flags = D3D12_EXPORT_FLAG_NONE;
    }

    // Create a library descriptor combining the DXIL code and the export names
    m_d3d_data->m_desc.DXILLibrary.BytecodeLength = m_dxil_library->GetBufferSize();
    m_d3d_data->m_desc.DXILLibrary.pShaderBytecode = m_dxil_library->GetBufferPointer();

    m_d3d_data->m_desc.NumExports = static_cast<UINT>(m_exported_symbols.size());
    m_d3d_data->m_desc.pExports = m_d3d_data->m_exports.data();
}

// ------------------------------------------------------------------------------------------------

std::vector<Shader_library> Shader_compiler::compile_shader_library(
    const Base_options* options,
    const std::string& file_name,
    const std::map<std::string, std::string>* defines,
    const std::vector<std::string>& entry_points)
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
        return {};
    }
    return compile_shader_library_from_string(
        options, shader_source, file_name, defines, entry_points);
}

// ------------------------------------------------------------------------------------------------

std::vector<Shader_library> Shader_compiler::compile_shader_library_from_string(
    const Base_options* options,
    const std::string& shader_source,
    const std::string& debug_name,
    const std::map<std::string, std::string>* defines,
    const std::vector<std::string>& entry_points)
{
    // compute virtual hlsl and debug file names
    std::string filename = mi::examples::io::normalize(debug_name);
    size_t p = filename.find_last_of('/');
    if (p != std::string::npos)
        filename = filename.substr(p + 1);
    filename = filename.substr(0, filename.find_last_of('.'));

    std::string dump_dir = mi::examples::io::get_working_directory() + "/dxc";
    std::string base_file_name = dump_dir + "/" + filename;

    std::wstring file_name_hlsl = mi::examples::strings::str_to_wstr(
        dump_dir + "/" + filename + ".hlsl");

    // we only need this directory when dumping shader files
    if (options->gpu_debug)
    {
        // store all intermediate debug files into a sub-folder
        mi::examples::io::mkdir(dump_dir);

        // dump hlsl code
        FILE* file = _wfopen(file_name_hlsl.c_str(), L"w");
        if (file)
        {
            fwrite(shader_source.c_str(), shader_source.size(), 1, file);
            fclose(file);
        }
    }

#ifdef MDL_ENABLE_SLANG
    if (options->use_slang)
    {
        return compile_shader_library_from_string_slang(
            options, shader_source, debug_name, defines, entry_points, base_file_name);
    }
#endif
    return compile_shader_library_from_string_dxc(
        options, shader_source, debug_name, defines, entry_points, base_file_name);
}

std::vector<Shader_library> Shader_compiler::compile_shader_library_from_string_dxc(
    const Base_options* options,
    const std::string& shader_source,
    const std::string& debug_name,
    const std::map<std::string, std::string>* defines,
    const std::vector<std::string>& entry_points,
    const std::string& base_file_name)
{
    // will be filled
    std::vector<Shader_library> dxil_libraries;

    // DXC compilation arguments
    std::vector<LPCWSTR> arguments;

    // include directory
    std::wstring inc = mi::examples::strings::str_to_wstr(
        mi::examples::io::get_executable_folder());
    arguments.push_back(L"-I");
    arguments.push_back(inc.c_str());

    // target profile
    arguments.push_back(L"-T");
    arguments.push_back(L"lib_6_3");

    // add debug symbols
    if (options->gpu_debug)
    {
        arguments.push_back(DXC_ARG_DEBUG); // -Zi
        // arguments.push_back(DXC_ARG_WARNINGS_ARE_ERRORS); //-WX
    }
    else
    {
        // arguments.push_back(DXC_ARG_SKIP_VALIDATION); // -Vd // will fail the pipeline creation
    }

    // optimization Od, O0, O1, O2, O3
    std::wstring optimization = mi::examples::strings::str_to_wstr("/" + options->shader_opt);
    arguments.push_back(optimization.c_str());

    // remove debug and reflection info from the actual shader blob
    // since we will have them in separate files (they are part of the compiler result anyway)
    arguments.push_back(L"-Qstrip_debug");
    arguments.push_back(L"-Qstrip_reflect");

    // since there are only a few defines, copying them seems okay
    std::vector<std::wstring> wstrings;
    if (defines) {
        for (const auto d : *defines) {
            arguments.push_back(L"-D");
            wstrings.push_back(mi::examples::strings::str_to_wstr(d.first + "=" + d.second));
            arguments.push_back(wstrings.back().c_str());
        }
    }

    ComPtr<IDxcCompiler3> compiler = nullptr;
    if (log_on_failure(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf())),
        "Failed to create IDxcCompiler", SRC))
        return {};

    ComPtr<IDxcUtils> utils = nullptr;
    if (log_on_failure(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(utils.GetAddressOf())),
        "Failed to create IDxcUtils", SRC))
        return {};

    // create a source blob
    ComPtr<IDxcBlobEncoding> shader_source_blob;
    if (log_on_failure(utils->CreateBlob(
        (LPBYTE)shader_source.c_str(), (uint32_t)shader_source.size(), CP_UTF8,
        shader_source_blob.GetAddressOf()),
        "Failed to create shader source blob: " + debug_name, SRC))
        return {};

    // create a special include handler to be able to run the applications with
    // different working directories
    IDxcIncludeHandler* base_handler;
    if (log_on_failure(utils->CreateDefaultIncludeHandler(&base_handler),
        "Failed to create Include Handler.", SRC))
        return {};
    ComPtr<IncludeHandler> include_handler = new IncludeHandler(base_handler);

    // compile the library
    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = shader_source_blob->GetBufferPointer();
    sourceBuffer.Size = shader_source_blob->GetBufferSize();
    sourceBuffer.Encoding = 0;
    ComPtr<IDxcResult> result;
    HRESULT hr;
    {
        Timing t("DXC Compile: " + debug_name);
        auto p = m_app->get_profiling().measure("DXC Compile: " + debug_name);

        hr = compiler->Compile(&sourceBuffer,
            arguments.data(), (UINT32)arguments.size(),
            include_handler.Get(), IID_PPV_ARGS(result.GetAddressOf()));
    }

    // check for errors
    result->GetStatus(&hr);
    ComPtr<IDxcBlobUtf8> error;
    if (log_on_failure(result->GetOutput(
        DXC_OUT_ERRORS, IID_PPV_ARGS(error.GetAddressOf()), nullptr),
        "Failed to get compilation error for source: " + debug_name, SRC))
        return {};
    if (error && error->GetStringLength())
    {
        std::vector<char> infoLog(error->GetBufferSize() + 1);
        memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
        infoLog[error->GetBufferSize()] = 0;

        // if the status is successful, the output only contains warnings
        if (SUCCEEDED(hr)) {
            std::string message = "Shader Compiler Warning: " + debug_name + "\n";
            message.append(infoLog.data());
            log_warning(message, SRC);
        } else {
            // otherwise, the output contains at least one error
            std::string message = "Shader Compiler Error: " + debug_name + "\n";
            message.append(infoLog.data());
            log_error(message, SRC);
        }
    }
    if (log_on_failure(hr, "Failed to compile shader source: " + debug_name, SRC))
        return {};

    // get the library DXIL code
    IDxcBlob* dxil_blob;
    if (log_on_failure(result->GetOutput(
        DXC_OUT_OBJECT, __uuidof(IDxcBlob), (void**)&dxil_blob, nullptr),
        "Failed to get shader blob for source: " + debug_name, SRC))
        return {};

    dxil_libraries.push_back(Shader_library(dxil_blob, entry_points));

    if (options->gpu_debug)
    {
        // get and write pdb
        ComPtr<IDxcBlob> debug_info_blob;
        IDxcBlobUtf16* debug_info_path;
        if (log_on_failure(result->GetOutput(
            DXC_OUT_PDB, IID_PPV_ARGS(debug_info_blob.GetAddressOf()), &debug_info_path),
            "Failed to get debug blob for source: " + debug_name, SRC))
            return {};

        std::wstring file_name_dbg = mi::examples::strings::str_to_wstr(base_file_name + ".dbg");
        FILE* file = _wfopen(file_name_dbg.c_str(), L"wb");
        if (file)
        {
            fwrite(debug_info_blob->GetBufferPointer(), debug_info_blob->GetBufferSize(), 1, file);
            fclose(file);
        }

        // write compiled shader library
        std::wstring file_name_dxil = mi::examples::strings::str_to_wstr(base_file_name + ".dxil");
        file = _wfopen(file_name_dxil.c_str(), L"wb");
        if (file)
        {
            fwrite(dxil_blob->GetBufferPointer(), dxil_blob->GetBufferSize(), 1, file);
            fclose(file);
        }

        // llvm code dump not available in dxcompiler library
        // so we print dxc arguments to a file to reproduce externally
        std::wstring file_name_hlsl = mi::examples::strings::str_to_wstr(base_file_name + ".hlsl");
        std::wstring file_name_ll = mi::examples::strings::str_to_wstr(base_file_name + ".ll");
        std::wstring dxc_command_line = L"";
        for (const auto& it : arguments)
        {
            if (dxc_command_line.empty())
                dxc_command_line = std::wstring(it);
            else
                dxc_command_line += L" " + std::wstring(it);
        }
        dxc_command_line += L" -Fc " + file_name_ll;
        dxc_command_line += L" " + file_name_hlsl;

        std::wstring file_name_ll_cmd_args = file_name_ll + L".cmd_args";
        file = _wfopen(file_name_ll_cmd_args.c_str(), L"w");
        if (file)
        {
            std::string dxc_command_line_a = mi::examples::strings::wstr_to_str(dxc_command_line);
            fwrite(dxc_command_line_a.c_str(), dxc_command_line_a.size(), 1, file);
            fclose(file);
        }

        log_verbose("DXC Command: " + mi::examples::strings::wstr_to_str(dxc_command_line));
    }

    // return the dxil shader blob
    return dxil_libraries;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

#ifdef MDL_ENABLE_SLANG
std::vector<Shader_library> Shader_compiler::compile_shader_library_from_string_slang(
    const Base_options* options,
    const std::string& shader_source,
    const std::string& debug_name,
    const std::map<std::string, std::string>* defines,
    const std::vector<std::string>& entry_points,
    const std::string& base_file_name)
{
    // will be filled
    std::vector<Shader_library> dxil_libraries;

    std::unique_ptr<SlangSession, void(*)(SlangSession*)> slang_session(
        g_slang_session_pool.acquire_session(),
        [](SlangSession* session) { g_slang_session_pool.release_session(session); }
    );

    std::string dxr_dir = mi::examples::io::get_executable_folder();
    slang_session->setDownstreamCompilerPath(SLANG_PASS_THROUGH_DXC, dxr_dir.c_str());

    Slang::ComPtr<slang::ICompileRequest> slang_compile_request;
    SlangResult result = slang_session->createCompileRequest(slang_compile_request.writeRef());
    if (SLANG_FAILED(result))
    {
        log_error("SlangSession::createCompileRequest failed with code "
            + std::to_string(result) + ": " + debug_name, SRC);
        return {};
    }

    std::string include_path = mi::examples::io::get_executable_folder();
    spAddSearchPath(slang_compile_request, include_path.c_str());

    spSetDebugInfoLevel(slang_compile_request, SLANG_DEBUG_INFO_LEVEL_NONE);
    spSetOptimizationLevel(slang_compile_request, shader_opt_to_slang_opt(options->shader_opt));

    SlangProfileID slang_profile = spFindProfile(slang_session.get(), "lib_6_3");
    if (slang_profile == SLANG_PROFILE_UNKNOWN)
    {
        log_error("spFindProfile failed to find a compatible profile for 'lib_6_3': " +
            debug_name, SRC);
        return {};
    }

    int target_index = spAddCodeGenTarget(slang_compile_request, SLANG_DXIL);
    int target_index_asm = 0;
    spSetTargetProfile(slang_compile_request, target_index, slang_profile);

    if (options->gpu_debug)
    {
        target_index_asm = spAddCodeGenTarget(slang_compile_request, SLANG_DXIL_ASM);
        spSetTargetProfile(slang_compile_request, target_index_asm, slang_profile);
    }
    //spSetPassThrough(slang_compile_request, SLANG_PASS_THROUGH_DXC);

    int unit_index = spAddTranslationUnit(
        slang_compile_request, SLANG_SOURCE_LANGUAGE_HLSL, nullptr);

    for (const auto& it : entry_points)
    {
        SlangStage stage = SLANG_STAGE_NONE;
        if (mi::examples::strings::contains(it.c_str(), "MissProgram"))
            stage = SLANG_STAGE_MISS;
        else if (mi::examples::strings::contains(it.c_str(), "AnyHitProgram"))
            stage = SLANG_STAGE_ANY_HIT;
        else if (mi::examples::strings::contains(it.c_str(), "ClosestHitProgram"))
            stage = SLANG_STAGE_CLOSEST_HIT;
        else if (mi::examples::strings::contains(it.c_str(), "RayGenProgram"))
            stage = SLANG_STAGE_RAY_GENERATION;
        if (stage == SLANG_STAGE_NONE)
        {
            log_error("Entrypoint name is not following naming convention': " + it, SRC);
            return {};
        }

        spAddEntryPoint(slang_compile_request, unit_index, it.c_str(), stage);
    }

    size_t sep_pos = base_file_name.rfind('/');
    std::string filepath = include_path + "/" + base_file_name.substr(sep_pos + 1) + ".hlsl";
    spAddTranslationUnitSourceString(
        slang_compile_request, unit_index, filepath.c_str(), shader_source.c_str());

    if (defines)
    {
        for (const auto d : *defines)
            spAddPreprocessorDefine(slang_compile_request, d.first.c_str(), d.second.c_str());
    }

    {
        Timing t("Slang Compile: " + debug_name);
        auto p = m_app->get_profiling().measure("Slang Compile: " + debug_name);

        result = spCompile(slang_compile_request);


        if (auto diagnostics = spGetDiagnosticOutput(slang_compile_request))
        {
            std::string errMsg(diagnostics);

            if (errMsg.size() > 0) {
                int line = -1;
                std::string msg = "";

                size_t lineLocBegin = errMsg.find_first_of('(');
                size_t lineLocEnd = errMsg.find_first_of(')');
                if (lineLocBegin != std::string::npos) {
                    std::string lineStr = errMsg.substr(
                        lineLocBegin + 1, lineLocEnd - lineLocBegin - 1);
                    msg = errMsg.substr(lineLocEnd + 3);

                    line = std::stoi(lineStr);
                }

                if (lineLocBegin != std::string::npos)
                    log_info(msg.c_str());
            }
        }

        if (SLANG_FAILED(result))
        {
            std::string message = "Shader Compiler Error: " + debug_name + "\n";
            message.append(spGetDiagnosticOutput(slang_compile_request));
            log_error(message, SRC);
            log_error("spCompile failed with code " + 
                std::to_string(result) + ": " + debug_name,SRC);
            return {};
        }
    }

    SlangReflection* slang_reflection = spGetReflection(slang_compile_request);
    SlangUInt entry_point_count = spReflection_getEntryPointCount(slang_reflection);

    for (int i = 0; i < entry_point_count; ++i)
    {
        size_t entry_point_code_size;
        spGetEntryPointCode(slang_compile_request, i, &entry_point_code_size);

        std::vector<char> code(entry_point_code_size);
        const void* entry_point_code = spGetEntryPointCode(
            slang_compile_request, i, &entry_point_code_size);

        // get the DXIL code block
        ISlangBlob* dxil_blob = nullptr;
        spGetEntryPointCodeBlob(slang_compile_request, i, target_index, &dxil_blob);
        SlangReflectionEntryPoint* entry_point_relf =
            spReflection_getEntryPointByIndex(slang_reflection, i);
        const char* entry_point_name = spReflectionEntryPoint_getName(entry_point_relf);
        dxil_libraries.push_back(Shader_library((IDxcBlob*)dxil_blob, { entry_point_name }));

        // get ll
        if (options->gpu_debug)
        {
            ISlangBlob* asm_blob = nullptr;
            spGetEntryPointCodeBlob(slang_compile_request, i, target_index_asm, &asm_blob);
            if (asm_blob)
            {
                std::wstring filename_ll = mi::examples::strings::str_to_wstr(
                    base_file_name + "_" + entry_point_name + ".ll");
                FILE* file = _wfopen(filename_ll.c_str(), L"wb");
                if (file)
                {
                    size_t buffer_size = asm_blob->getBufferSize();
                    fwrite(asm_blob->getBufferPointer(), buffer_size, 1, file);
                    fclose(file);
                }
            }
            asm_blob->release();
        }
    }

    return dxil_libraries;
}
#endif // MDL_ENABLE_SLANG

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
    : m_shader_id(nullptr)
    , m_mapped_table_pointer(nullptr)
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

    m_binding_table_buffer_latest_requested_state = D3D12_RESOURCE_STATE_COMMON;
    auto heap_properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_NONE, &desc,
        m_binding_table_buffer_latest_requested_state, nullptr,
        IID_PPV_ARGS(&m_binding_table_buffer)),
        "Failed to create buffer for shader binding table: " + m_debug_name, SRC))
        return false;
    set_debug_name(m_binding_table_buffer.Get(), m_debug_name);

    m_binding_table_buffer_upload_latest_requested_state = 
        D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_COPY_SOURCE;
    heap_properties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    if (log_on_failure(m_app->get_device()->CreateCommittedResource(
        &heap_properties, D3D12_HEAP_FLAG_NONE, &desc,
        m_binding_table_buffer_upload_latest_requested_state, nullptr,
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

    // transition the resources on first usage
    if (m_binding_table_buffer_upload_latest_requested_state == D3D12_RESOURCE_STATE_COMMON)
    {
        auto rb = CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer_upload.Get(),
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_GENERIC_READ);
        command_list->ResourceBarrier(1, &rb);
        m_binding_table_buffer_upload_latest_requested_state = D3D12_RESOURCE_STATE_GENERIC_READ;
    }
    if (m_binding_table_buffer_latest_requested_state == D3D12_RESOURCE_STATE_COMMON)
    {
        auto rb = CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer.Get(),
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        command_list->ResourceBarrier(1, &rb);
        m_binding_table_buffer_latest_requested_state = 
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    auto rb = CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->ResourceBarrier(1, &rb);
    m_binding_table_buffer_latest_requested_state = D3D12_RESOURCE_STATE_COPY_DEST;

    command_list->CopyResource(
        m_binding_table_buffer.Get(), m_binding_table_buffer_upload.Get());

    rb = CD3DX12_RESOURCE_BARRIER::Transition(m_binding_table_buffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    command_list->ResourceBarrier(1, &rb);
    m_binding_table_buffer_latest_requested_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
}

// ------------------------------------------------------------------------------------------------

D3D12_DISPATCH_RAYS_DESC Shader_binding_tables::get_dispatch_description() const
{
    return m_prefilled_dispatch_description;
}

}}} // mi::examples::mdl_d3d12
