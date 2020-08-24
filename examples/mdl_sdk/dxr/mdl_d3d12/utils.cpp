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

#include "utils.h"
#include <Windows.h>
#include <d3d12.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <atomic>
#include <utils/strings.h>

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

// ------------------------------------------------------------------------------------------------

void print(
    const std::string& prefix,
    const std::string& message,
    const std::string& file,
    int line)
{
    std::string m = prefix;
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

std::string print_nested_exceotion(const std::exception& e)
{
    std::string message = e.what();
    try {
        std::rethrow_if_nested(e);
    }
    catch (const std::exception& nested) {
        message += "\n               nested: " + print_nested_exceotion(nested);
    }
    return message;
}

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void log_verbose(const std::string& message, const std::string& file, int line)
{
    print("[MDL_D3D12] [VERBOSE] ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_info(const std::string& message, const std::string& file, int line)
{
    print("[MDL_D3D12] [INFO]    ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_warning(const std::string& message, const std::string& file, int line)
{
    print("[MDL_D3D12] [WARNING] ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(const std::string& message, const std::string& file, int line)
{
    print("[MDL_D3D12] [ERROR]   ", message, file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(
    const std::string& message,
    const std::exception& exception,
    const std::string& file, int line)
{
    print("[MDL_D3D12] [ERROR]   ",
        message + " " + print_nested_exceotion(exception), file, line);
}

// ------------------------------------------------------------------------------------------------

void log_error(const std::exception& exception, const std::string& file, int line)
{
    print("[MDL_D3D12] [ERROR]   ", print_nested_exceotion(exception), file, line);
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
// ------------------------------------------------------------------------------------------------

Timing::Timing(std::string operation)
    : m_operation(operation)
{
    m_start = std::chrono::high_resolution_clock::now();
}

// ------------------------------------------------------------------------------------------------

Timing::~Timing()
{
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = (stop - m_start).count() * 1e-9;
    log_info("Finished '" + m_operation + "' after " +
                std::to_string(elapsed_seconds) + " seconds.");
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Profiling::Measurement::Measurement(Profiling* p, std::string operation)
    : m_profiling(p)
    , m_operation(operation)
{
    m_start = std::chrono::high_resolution_clock::now();
}

// ------------------------------------------------------------------------------------------------

Profiling::Measurement::~Measurement()
{
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = (stop - m_start).count() * 1e-9;
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

}}} // mi::examples::mdl_d3d12
