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

// examples/mdl_sdk/dxr/mdl_d3d12/mdl_d3d12.h

#ifndef MDL_D3D12_COMMON_H
#define MDL_D3D12_COMMON_H

#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <D3d12.h>
#include <d3dx12.h>
#include <dxcapi.h>
#include <dxgi1_4.h>
#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <D3d12SDKLayers.h>
#include <wrl.h>
#include <Windows.h>

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <set>
#include <vector>
#include <stack>
#include <deque>
#include <functional>
#include <mutex>
#include <atomic>

#include "utils.h"

// mdl example shared
#include <utils/enums.h>
#include <utils/io.h>
#include <utils/os.h>
#include <utils/strings.h>

namespace mi { namespace examples { namespace mdl_d3d12
{
    using namespace DirectX;
    template<class T>
    using ComPtr = Microsoft::WRL::ComPtr<T>;

    typedef ID3D12Device5 D3DDevice;
    typedef ID3D12GraphicsCommandList4 D3DCommandList;

    class Descriptor_heap;

    /// Identifies on which heap and which index a resource view is located.
    /// A simple integer could do as well, but this is more explicit.
    struct Descriptor_heap_handle
    {
        friend class Descriptor_heap;

        explicit Descriptor_heap_handle();
        virtual ~Descriptor_heap_handle() = default;

        bool is_valid() const { return m_descriptor_heap != 0; }
        size_t get_heap_index() const { return m_index; }

        operator size_t() const { return m_index; }

        Descriptor_heap_handle create_offset(size_t offset);

        /// Get the internal D3D CPU descriptor handle
        D3D12_CPU_DESCRIPTOR_HANDLE get_cpu_handle() const;
        /// Get the internal D3D GPU descriptor handle
        D3D12_GPU_DESCRIPTOR_HANDLE get_gpu_handle() const;

    private:
        explicit Descriptor_heap_handle(Descriptor_heap* heap, size_t index);
        Descriptor_heap* m_descriptor_heap;
        size_t m_index;
    };

    /// Shared base class for all textures and buffers.
    class Resource
    {
    public:
        virtual ~Resource() = default;
        virtual std::string get_debug_name() const = 0;
    };

}}} // mi::examples::mdl_d3d12
#endif
