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

// examples/mdl_sdk/dxr/mdl_d3d12/utils.h

#ifndef MDL_D3D12_UTILS_H
#define MDL_D3D12_UTILS_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <Windows.h>
#include <DirectXMath.h>

#define SRC __FILE__,__LINE__

struct ID3D12Object;

namespace mi { namespace examples { namespace mdl_d3d12
{
    const float PI = 3.14159265358979323846f;
    const float PI_OVER_2 = PI * 0.5f;
    const float ONE_OVER_PI = 0.318309886183790671538f;
    const float SQRT_2 = 1.41421356237f;
    const float SQRT_3 = 1.73205080757f;

    void log_verbose(
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    void log_info(
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    void log_warning(
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    void log_error(
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    void log_error(
        const std::exception& exception,
        const std::string& file,
        int line = 0);

    void log_error(
        const std::string& message,
        const std::exception& exception,
        const std::string& file, int line);

    bool log_on_failure(
        HRESULT error_code,
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    void throw_on_failure(
        HRESULT error_code,
        const std::string& message,
        const std::string& file = "",
        int line = 0);

    // flushes all ongoing async logging tasks. has to be called before closing the app.
    void flush_loggers();

    // set a log file to use or \c nullptr in order disable file logging when active.
    // has to be called with a nullptr before closing the app.
    void log_set_file_path(const char* log_file_path);

    void set_debug_name(ID3D12Object* obj, const std::string& name);

    inline size_t round_to_power_of_two(size_t value, size_t power_of_two_factor)
    {
        return (value + (power_of_two_factor - 1)) & ~(power_of_two_factor - 1);
    }

    // --------------------------------------------------------------------------------------------

    struct Timing
    {
        explicit Timing(std::string operation);
        ~Timing();

    private:
        std::string m_operation;
        std::chrono::steady_clock::time_point m_start;
    };

    // --------------------------------------------------------------------------------------------

    class Profiling
    {
    public:
        struct Measurement
        {
            ~Measurement();
        private:
            friend class Profiling;
            explicit Measurement(Profiling* p, std::string operation);

            Profiling* m_profiling;
            std::string m_operation;
            std::chrono::steady_clock::time_point m_start;
        };

        Measurement measure(std::string operation);
        void print_statistics() const;
        void reset_statistics() { m_statistics.clear(); }

    private:
        struct Entry
        {
            double average;
            size_t count;
        };

        void on_measured(const Measurement& m, const double& value);

        std::map<std::string, Entry> m_statistics;
        std::mutex m_statistics_mtx;
    };

    // --------------------------------------------------------------------------------------------
    // DirectX Math
    // --------------------------------------------------------------------------------------------

    inline DirectX::XMMATRIX inverse(const DirectX::XMMATRIX& m, DirectX::XMVECTOR* determinants = nullptr)
    {
        if (determinants)
            return DirectX::XMMatrixInverse(determinants, m);

        DirectX::XMVECTOR det;
        return DirectX::XMMatrixInverse(&det, m);
    }

    inline DirectX::XMFLOAT3 normalize(const DirectX::XMFLOAT3& v)
    {
        float inv_length = 1.0f / std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        return {v.x * inv_length, v.y * inv_length, v.z * inv_length};
    }

    inline float length(const DirectX::XMFLOAT3& v)
    {
        return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    inline float length2(const DirectX::XMFLOAT3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    inline float dot(const DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    inline float average(const DirectX::XMFLOAT3& v)
    {
        return (v.x + v.y + v.z) * 0.33333333333333f;
    }

    inline float maximum(const DirectX::XMFLOAT3& v)
    {
        return std::max<float>(v.x, std::max<float>(v.y, v.z));
    }

    inline float minimum(const DirectX::XMFLOAT3& v)
    {
        return std::min<float>(v.x, std::min<float>(v.y, v.z));
    }

    inline DirectX::XMFLOAT3 cross(const DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        DirectX::XMFLOAT3 res;
        res.x = (v1.y * v2.z) - (v1.z * v2.y);
        res.y = (v1.z * v2.x) - (v1.x * v2.z);
        res.z = (v1.x * v2.y) - (v1.y * v2.x);
        return res;
    }

    inline DirectX::XMFLOAT3 operator-(const DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        DirectX::XMFLOAT3 res;
        res.x = v1.x - v2.x;
        res.y = v1.y - v2.y;
        res.z = v1.z - v2.z;
        return res;
    }

    inline DirectX::XMFLOAT3 operator+(const DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        DirectX::XMFLOAT3 res;
        res.x = v1.x + v2.x;
        res.y = v1.y + v2.y;
        res.z = v1.z + v2.z;
        return res;
    }

    inline DirectX::XMFLOAT3 operator*(const DirectX::XMFLOAT3& v1, float s)
    {
        DirectX::XMFLOAT3 res;
        res.x = v1.x * s;
        res.y = v1.y * s;
        res.z = v1.z * s;
        return res;
    }

    inline void operator*=(DirectX::XMFLOAT3& v, float s)
    {
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }

    inline void operator/=(DirectX::XMFLOAT3& v, float s)
    {
        v.x /= s;
        v.y /= s;
        v.z /= s;
    }

    inline DirectX::XMFLOAT3 operator-(const DirectX::XMFLOAT3& v1)
    {
        DirectX::XMFLOAT3 res;
        res.x = -v1.x;
        res.y = -v1.y;
        res.z = -v1.z;
        return res;
    }

    inline void operator+=(DirectX::XMFLOAT3& v1, const DirectX::XMFLOAT3& v2)
    {
        v1.x += v2.x;
        v1.y += v2.y;
        v1.z += v2.z;
    }

}}} // mi::examples::mdl_d3d12
#endif
