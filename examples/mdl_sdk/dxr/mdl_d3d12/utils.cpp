/******************************************************************************
 * Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "utils.h"
#include <Windows.h>
#include <wrl.h>
#include <d3d12.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <set>
#include <atomic>
#include <utils/strings.h>
#include <chrono>
#include <TlHelp32.h>
#include <processthreadsapi.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{

static std::atomic<size_t> s_cerr_active_counter = 0;
static std::mutex s_cerr_mtx;

static std::atomic<size_t> s_vs_console_active_counter = 0;
static std::mutex s_vs_console_mtx;

static std::atomic<size_t> s_log_file_active_counter = 0;
static std::mutex s_log_file_mtx;
static std::ofstream s_log_file;

static Log_level s_log_level = Log_level::Info;

template<class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;
static ID3D12Device* s_dred_device;

// ------------------------------------------------------------------------------------------------

std::string current_time()
{
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    char text[12] = { '\0' };
    auto res = std::strftime(text, sizeof(text), "[%H:%M:%S] ", std::localtime(&now));
    return text;
}

// ------------------------------------------------------------------------------------------------



void print(
    const std::string& prefix,
    const std::string& message,
    const std::string& file,
    int line)
{
    std::string m = current_time();
    m.append(prefix);
    m.append(message);
    if (!file.empty())
    {
        m.append("\n                file: ");
        m.append(file);
        m.append(", line: ");
        m.append(std::to_string(line));
        m.append("\n");
    }
    else
    {
        m.append("\n");
    }

    s_cerr_active_counter++;
    std::thread task_cerr([m]() {
        std::lock_guard<std::mutex> lock(s_cerr_mtx);
        std::cerr << m.c_str();
        s_cerr_active_counter--;
    });
    task_cerr.detach();

    s_vs_console_active_counter++;
    std::thread task_vs_console([m]() {
        std::lock_guard<std::mutex> lock(s_vs_console_mtx);
        OutputDebugStringA(m.c_str());
        s_vs_console_active_counter--;
    });
    task_vs_console.detach();

    if (s_log_file.is_open())
    {
        s_log_file_active_counter++;
        std::thread task_log_file([m]() {
            std::lock_guard<std::mutex> lock(s_log_file_mtx);
            s_log_file << m.c_str();
            s_log_file_active_counter--;
        });
        task_log_file.detach();
    }
}


// ------------------------------------------------------------------------------------------------

std::string to_string(HRESULT error_code)
{
    switch (error_code)
    {
        case DXGI_ERROR_INVALID_CALL:
            return "DXGI_ERROR_INVALID_CALL";
        case DXGI_ERROR_WAS_STILL_DRAWING:
            return "DXGI_ERROR_WAS_STILL_DRAWING";
        case DXGI_ERROR_DEVICE_REMOVED:
            return "DXGI_ERROR_DEVICE_REMOVED";
        case DXGI_ERROR_DEVICE_HUNG:
            return "DXGI_ERROR_DEVICE_HUNG";
        case DXGI_ERROR_DEVICE_RESET:
            return "DXGI_ERROR_DEVICE_RESET";
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
            return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
        case E_FAIL:
            return "E_FAIL";
        case E_INVALIDARG:
            return "E_INVALIDARG";
        case E_OUTOFMEMORY:
            return "E_OUTOFMEMORY";
        case E_NOTIMPL:
            return "E_NOTIMPL";
        case S_FALSE:
            return "S_FALSE";
        case S_OK:
            return "S_OK";
        default:
            return "";
    }
}

// ------------------------------------------------------------------------------------------------

std::string to_string(D3D12_AUTO_BREADCRUMB_OP op)
{
    switch (op)
    {
    case D3D12_AUTO_BREADCRUMB_OP_SETMARKER:
        return "SETMARKER";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT:
        return "BEGINEVENT";
    case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT:
        return "ENDEVENT";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED:
        return "DRAWINSTANCED";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED:
        return "DRAWINDEXEDINSTANCED";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT:
        return "EXECUTEINDIRECT";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCH:
        return "DISPATCH";
    case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION:
        return "COPYBUFFERREGION";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION:
        return "COPYTEXTUREREGION";
    case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE:
        return "COPYRESOURCE";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTILES:
        return "COPYTILES";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE:
        return "RESOLVESUBRESOURCE";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW:
        return "CLEARRENDERTARGETVIEW";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW:
        return "CLEARUNORDEREDACCESSVIEW";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW:
        return "CLEARDEPTHSTENCILVIEW";
    case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER:
        return "RESOURCEBARRIER";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE:
        return "EXECUTEBUNDLE";
    case D3D12_AUTO_BREADCRUMB_OP_PRESENT:
        return "PRESENT";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA:
        return "RESOLVEQUERYDATA";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION:
        return "BEGINSUBMISSION";
    case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION:
        return "ENDSUBMISSION";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME:
        return "DECODEFRAME";
    case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES:
        return "PROCESSFRAMES";
    case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT:
        return "ATOMICCOPYBUFFERUINT";
    case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT64:
        return "ATOMICCOPYBUFFERUINT64";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCEREGION:
        return "RESOLVESUBRESOURCEREGION";
    case D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE:
        return "WRITEBUFFERIMMEDIATE";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME1:
        return "DECODEFRAME1";
    case D3D12_AUTO_BREADCRUMB_OP_SETPROTECTEDRESOURCESESSION:
        return "SETPROTECTEDRESOURCESESSION";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME2:
        return "DECODEFRAME2";
    case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES1:
        return "PROCESSFRAMES1";
    case D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE:
        return "BUILDRAYTRACINGACCELERATIONSTRUCTURE";
    case D3D12_AUTO_BREADCRUMB_OP_EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO:
        return "EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO";
    case D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE:
        return "COPYRAYTRACINGACCELERATIONSTRUCTURE";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS:
        return "DISPATCHRAYS";
    case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEMETACOMMAND:
        return "INITIALIZEMETACOMMAND";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEMETACOMMAND:
        return "EXECUTEMETACOMMAND";
    case D3D12_AUTO_BREADCRUMB_OP_ESTIMATEMOTION:
        return "ESTIMATEMOTION";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVEMOTIONVECTORHEAP:
        return "RESOLVEMOTIONVECTORHEAP";
    case D3D12_AUTO_BREADCRUMB_OP_SETPIPELINESTATE1:
        return "SETPIPELINESTATE1";

    // available from 10.0.18362.0
    #if WDK_NTDDI_VERSION > NTDDI_WIN10_RS5
        case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEEXTENSIONCOMMAND:
            return "INITIALIZEEXTENSIONCOMMAND";
        case D3D12_AUTO_BREADCRUMB_OP_EXECUTEEXTENSIONCOMMAND:
            return "EXECUTEEXTENSIONCOMMAND";
    #endif

    default:
        return "";
    }
}

// ------------------------------------------------------------------------------------------------

std::string evaluate_dred()
{
    if (!s_dred_device)
        return "";

    // see https://docs.microsoft.com/en-us/windows/win32/direct3d12/use-dred
    // for more information

    std::string output = "";
    // available from 10.0.18362.0
    #if WDK_NTDDI_VERSION > NTDDI_WIN10_RS5

        ComPtr<ID3D12DeviceRemovedExtendedData> pDred;
        SUCCEEDED(s_dred_device->QueryInterface(IID_PPV_ARGS(&pDred)));
        
        const size_t max_nodes_to_print = 128; // arbitrary

        // output breadcrumbs data
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT DredAutoBreadcrumbsOutput;
        if (SUCCEEDED(pDred->GetAutoBreadcrumbsOutput(&DredAutoBreadcrumbsOutput)))
        {
            size_t current_node_i = 0;
            const D3D12_AUTO_BREADCRUMB_NODE* current =
                DredAutoBreadcrumbsOutput.pHeadAutoBreadcrumbNode;

            output += "\nBreadcrumb Nodes:";
            while (current && current_node_i < max_nodes_to_print)
            {
                UINT32 bc_count = current->BreadcrumbCount;
                UINT32 last_value = *current->pLastBreadcrumbValue;
                bool crashed = bc_count != last_value;

                output += mi::examples::strings::format("\n[%03d] Node has %d breadcrumbs.",
                    current_node_i, bc_count);

                // ring buffer of size 65536, rest is lost
                for (UINT32 b = 0; b < (bc_count % 65536); ++b)
                {
                    output += mi::examples::strings::format("\n      %-5d %-55s %s",
                        b,
                        to_string(current->pCommandHistory[b]).c_str(),
                        crashed ? (b < last_value ? "COMPLETED" : "NOT COMPLETE (probably)") : "");
                }
                current = current->pNext;
                current_node_i++;
            }
        }

        // output page fault data
        D3D12_DRED_PAGE_FAULT_OUTPUT DredPageFaultOutput;
        if (SUCCEEDED(pDred->GetPageFaultAllocationOutput(&DredPageFaultOutput)))
        {
            output += mi::examples::strings::format("\nPage Fault Virtual Address: 0x%X",
                DredPageFaultOutput.PageFaultVA);

            size_t current_node_i = 0;
            const D3D12_DRED_ALLOCATION_NODE* current =
                DredPageFaultOutput.pHeadExistingAllocationNode;

            output += "\nExisting DRED Allocation Nodes:";
            while (current && current_node_i < max_nodes_to_print)
            {
                output += mi::examples::strings::format("\n[%03d][type=%d] %s",
                    current_node_i, current->AllocationType, current->ObjectNameA);
                output += current->ObjectNameA;

                current = current->pNext;
                current_node_i++;
            }

            current_node_i = 0;
            current = DredPageFaultOutput.pHeadRecentFreedAllocationNode;

            output += "\nRecent Freed DRED Allocation Nodes:";
            while (current && current_node_i < max_nodes_to_print)
            {
                output += mi::examples::strings::format("\n[%03d][type=%d] %s",
                    current_node_i, current->AllocationType, current->ObjectNameA);
                output += current->ObjectNameA;

                current = current->pNext;
                current_node_i++;
            }
        }

        // printing this once is enough
        s_dred_device = nullptr;

    #endif

    return output;
}

// ------------------------------------------------------------------------------------------------

std::string print_nested_exception(const std::exception& e)
{
    std::string message = e.what();
    try {
        std::rethrow_if_nested(e);
    }
    catch (const std::exception& nested) {
        message += "\n               nested: " + print_nested_exception(nested);
    }
    return message;
}

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void set_log_level(Log_level level)
{
    s_log_level = level;
}

// ------------------------------------------------------------------------------------------------

void log_verbose(const std::string& message, const std::string& file, int line)
{
    if (static_cast<char>(s_log_level) < static_cast<char>(Log_level::Verbose))
        return;

    print("[MDL_D3D12] [VERBOSE] ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_info(const std::string& message, const std::string& file, int line)
{
    if(static_cast<char>(s_log_level) < static_cast<char>(Log_level::Info))
        return;

    print("[MDL_D3D12] [INFO]    ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_warning(const std::string& message, const std::string& file, int line)
{
    if (static_cast<char>(s_log_level) < static_cast<char>(Log_level::Warning))
        return;

    print("[MDL_D3D12] [WARNING] ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(const std::string& message, const std::string& file, int line)
{
    if (static_cast<char>(s_log_level) < static_cast<char>(Log_level::Error))
        return;

    print("[MDL_D3D12] [ERROR]   ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(
    const std::string& message,
    const std::exception& exception,
    const std::string& file, int line)
{
    print("[MDL_D3D12] [ERROR]   ",
        message + " " + print_nested_exception(exception), file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(const std::exception& exception, const std::string& file, int line)
{
    print("[MDL_D3D12] [ERROR]   ", print_nested_exception(exception), file, line);
}

// ------------------------------------------------------------------------------------------------

bool log_on_failure(
    HRESULT error_code, const std::string& message, const std::string& file, int line)
{
    if (SUCCEEDED(error_code))
        return false;

    std::string readable_error = to_string(error_code);
    if (!readable_error.empty())
        readable_error += " (" + std::to_string(error_code) + ")";
    else
        readable_error = std::to_string(error_code);

    if (error_code == DXGI_ERROR_DEVICE_REMOVED || error_code == DXGI_ERROR_DEVICE_RESET)
    {
        readable_error += "\n" + evaluate_dred() + "\n";
    }

    print("[MDL_D3D12] [FAILURE] ",
            message + "\n                     return code: " + readable_error,
            file, line);
    return true;
}

// ------------------------------------------------------------------------------------------------

void throw_on_failure(
    HRESULT error_code, const std::string& message, const std::string& file, int line)
{
    if (log_on_failure(error_code, message, file, line))
        throw(message.c_str());
}

// ------------------------------------------------------------------------------------------------

void flush_loggers()
{
    using namespace std::chrono_literals;
    {
        while (s_cerr_active_counter > 0)
            std::this_thread::sleep_for(1ms);

        std::lock_guard<std::mutex> lock(s_cerr_mtx);
        std::cerr << std::flush;
    }
    {
        while (s_vs_console_active_counter > 0)
            std::this_thread::sleep_for(1ms);

        std::lock_guard<std::mutex> lock(s_vs_console_mtx);
        // no flush needed but make sure to log is released before closing the app
    }
    if (s_log_file.is_open())
    {
        while (s_log_file_active_counter > 0)
            std::this_thread::sleep_for(1ms);

        std::lock_guard<std::mutex> lock(s_log_file_mtx);
        s_log_file << std::flush;
    }
}

// ------------------------------------------------------------------------------------------------

void log_set_file_path(const char* log_file_path)
{
    std::lock_guard<std::mutex> lock(s_log_file_mtx);

    // close current log file
    if (s_log_file.is_open())
        s_log_file.close();

    // open new log file
    if (log_file_path)
    {
        s_log_file = std::ofstream();
        s_log_file.open(log_file_path, std::ofstream::out | std::ofstream::trunc);

        if (!s_log_file.is_open())
            log_error("Failed to open log file for writing: " +
                std::string(log_file_path), SRC);

        s_log_file.seekp(std::ios_base::beg);
    }
}

// ------------------------------------------------------------------------------------------------

void set_debug_name(ID3D12Object* obj, const std::string& name)
{
    obj->SetName(mi::examples::strings::str_to_wstr(name).c_str());
}

// ------------------------------------------------------------------------------------------------

void set_dred_device(ID3D12Device* device)
{
    s_dred_device = device;
}

// ------------------------------------------------------------------------------------------------

Timing::Timing(std::string operation)
    : m_operation(operation)
{
    m_start = std::chrono::steady_clock::now();
}

// ------------------------------------------------------------------------------------------------

Timing::~Timing()
{
    auto stop = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(stop - m_start).count();
    log_info("Finished '" + m_operation + "' after " +
                std::to_string(elapsed_seconds) + " seconds.");
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Profiling::Measurement::Measurement(Profiling* p, std::string operation)
    : m_profiling(p)
    , m_operation(operation)
{
    m_start = std::chrono::steady_clock::now();
}

// ------------------------------------------------------------------------------------------------

Profiling::Measurement::~Measurement()
{
    auto stop = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(stop - m_start).count();
    m_profiling->on_measured(*this, elapsed_seconds);
}

// ------------------------------------------------------------------------------------------------

Profiling::Measurement Profiling::measure(std::string operation)
{
    return Measurement(this, operation);
}

// ------------------------------------------------------------------------------------------------

void Profiling::on_measured(const Measurement& m, const double& value)
{
    std::lock_guard<std::mutex> lock(m_statistics_mtx);
    auto found = m_statistics.find(m.m_operation);
    if (found == m_statistics.end())
        m_statistics[m.m_operation] = { value, 1 };
    else
    {
        size_t n = ++found->second.count;
        double one_over_n = 1.0 / double(n);
        found->second.average =
            found->second.average * double(n - 1) * one_over_n + value * one_over_n;
    }
}

// ------------------------------------------------------------------------------------------------

void Profiling::print_statistics() const
{
    std::string msg = "Profiling Statistics:";
    for (const auto& pair : m_statistics)
    {
        msg += mi::examples::strings::format("\n    %-60s %10.3fs (n = %d)",
            pair.first.c_str(), pair.second.average, pair.second.count);
    }
    log_info(msg);
}

// ------------------------------------------------------------------------------------------------

void Diagnostics::list_loaded_libraries()
{
    DWORD process_id = GetCurrentProcessId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPALL, process_id);
    MODULEENTRY32W module;
    memset(&module, 0, sizeof(MODULEENTRY32W));
    module.dwSize = sizeof(MODULEENTRY32W);
    bool has_next;
    DWORD last_error = ERROR_BAD_LENGTH;
    while (last_error == ERROR_BAD_LENGTH)
    {
        SetLastError(0);
        has_next = Module32FirstW(snapshot, &module);
        last_error = GetLastError();
        if (last_error == ERROR_NO_MORE_FILES)
            break;
    }

    std::set<std::string> module_list; // use a set for sorting
    while (has_next)
    {
        std::string path = mi::examples::strings::wstr_to_str(module.szExePath);
        std::replace(path.begin(), path.end(), '\\', '/');
        module_list.insert(path);
        has_next = Module32NextW(snapshot, &module);
    }

    std::string msg = "Loaded libraries:";
    for(auto& e : module_list)
        msg += mi::examples::strings::format("\n    %s", e.c_str());

    log_verbose(msg);
}

}}} // mi::examples::mdl_d3d12
