/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

namespace
{
    static std::mutex s_cerr_mtx;
    static std::mutex s_vs_console_mtx;
    static std::mutex s_log_file_mtx;
    static std::ofstream s_log_file;

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

        std::thread task_cerr([&, m]() {
            s_cerr_mtx.lock();
            std::cerr << m.c_str(); 
            s_cerr_mtx.unlock();
        });
        task_cerr.detach();

        std::thread task_vs_console([&, m]() {
            s_vs_console_mtx.lock();
            OutputDebugStringA(m.c_str());
            s_vs_console_mtx.unlock();
            });
        task_vs_console.detach();
        
        if (s_log_file.is_open())
        {
            std::thread task_log_file([&, m]() {
                s_log_file_mtx.lock();
                s_log_file << m.c_str();
                s_log_file_mtx.unlock();
                });
            task_log_file.detach();
        }
    }

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

namespace mdl_d3d12
{
    void log_info(const std::string& message, const std::string& file, int line)
    {
        print("[MDL_D3D12] [INFO]    ", message, file, line);
    }

    void log_warning(const std::string& message, const std::string& file, int line)
    {
        print("[MDL_D3D12] [WARNING] ", message, file, line);
    }

    void log_error(const std::string& message, const std::string& file, int line)
    {
        print("[MDL_D3D12] [ERROR]   ", message, file, line);
    }

    void log_error(const std::exception& exception, const std::string& file, int line)
    {
        print("[MDL_D3D12] [ERROR]   ", print_nested_exceotion(exception), file, line);
    }


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

    void throw_on_failure(
        HRESULT error_code, const std::string& message, const std::string& file, int line)
    {
        if (log_on_failure(error_code, message, file, line))
            throw(message.c_str());
    }


    void log_set_file_path(const char* log_file_path)
    {
        s_log_file_mtx.lock();

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

        s_log_file_mtx.unlock();
    }

    void set_debug_name(ID3D12Object* obj, const std::string& name)
    {
        obj->SetName(str_to_wstr(name).c_str());
    }


    Timing::Timing(std::string operation)
        : m_operation(operation)
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

    Timing::~Timing()
    {
        auto stop = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = (stop - m_start).count() * 1e-9;
        log_info("Finished '" + m_operation + "' after " + 
                 std::to_string(elapsed_seconds) + " seconds.");
    }
}
