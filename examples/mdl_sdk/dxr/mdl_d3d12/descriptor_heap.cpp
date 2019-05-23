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

#include "descriptor_heap.h"
#include "raytracing_pipeline.h"
#include "texture.h"

namespace mdl_d3d12
{
    // --------------------------------------------------------------------------------------------

    Descriptor_heap_handle::Descriptor_heap_handle()
        : m_descriptor_heap(nullptr)
        , m_index(static_cast<size_t>(-1)) 
    {
    }

    Descriptor_heap_handle::Descriptor_heap_handle(Descriptor_heap* heap, size_t index)
        : m_descriptor_heap(heap)
        , m_index(index) 
    {
    }

    // --------------------------------------------------------------------------------------------

    Descriptor_heap::Descriptor_heap(
        Base_application* app, 
        D3D12_DESCRIPTOR_HEAP_TYPE type, 
        size_t size, 
        std::string debug_name)

        : m_app(app)
        , m_debug_name(debug_name)
        , m_type(type)
        , m_size(size)
        , m_element_size(app->get_device()->GetDescriptorHandleIncrementSize(type))
        , m_cpu_heap_start{0}
        , m_gpu_heap_start{0}
    {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
        heap_desc.NumDescriptors = static_cast<UINT>(size);
        heap_desc.Type = m_type;

        if(m_type != D3D12_DESCRIPTOR_HEAP_TYPE_RTV && m_type != D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
 
        if (log_on_failure(
            m_app->get_device()->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&m_heap)),
            "Failed to create descriptor heap for texture.", SRC))
            return;
        set_debug_name(m_heap.Get(), m_debug_name);

        m_cpu_heap_start = m_heap->GetCPUDescriptorHandleForHeapStart();
        m_gpu_heap_start = m_heap->GetGPUDescriptorHandleForHeapStart();
    }

    Descriptor_heap::~Descriptor_heap()
    {
    }

    Descriptor_heap_handle Descriptor_heap::add_empty_view()
    {
        Descriptor_heap_handle handle(this, m_entries.size());
        Entry entry;
        m_entries.push_back(std::move(entry));
        return std::move(handle);
    }

    Descriptor_heap_handle Descriptor_heap::add_shader_resource_view(
        const D3D12_SHADER_RESOURCE_VIEW_DESC& desc, 
        ID3D12Resource* resource)
    {
        m_app->get_device()->CreateShaderResourceView(
            resource, &desc, get_cpu_handle(m_entries.size()));

        Descriptor_heap_handle handle(this, m_entries.size());
        Entry entry;

        m_entries.push_back(std::move(entry));
        return std::move(handle);
    }

    void Descriptor_heap::replace_by_shader_resource_view(
        const D3D12_SHADER_RESOURCE_VIEW_DESC& desc,
        ID3D12Resource* resource,
        const Descriptor_heap_handle& handle)
    {
        m_app->get_device()->CreateShaderResourceView(resource, &desc, get_cpu_handle(handle));
    }

    Descriptor_heap_handle Descriptor_heap::add_constant_buffer_view(
        const D3D12_CONSTANT_BUFFER_VIEW_DESC& desc)
    {
        m_app->get_device()->CreateConstantBufferView(&desc, get_cpu_handle(m_entries.size()));
        Descriptor_heap_handle handle(this, m_entries.size());
        Entry entry;

        m_entries.push_back(std::move(entry));
        return std::move(handle);
    }

    void Descriptor_heap::replace_by_constant_buffer_view(
        const D3D12_CONSTANT_BUFFER_VIEW_DESC & desc, 
        const Descriptor_heap_handle& handle)
    {
        m_app->get_device()->CreateConstantBufferView(&desc, get_cpu_handle(handle));
    }

    Descriptor_heap_handle Descriptor_heap::add_unordered_access_view(
        const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc, 
        ID3D12Resource* resource, 
        ID3D12Resource* counter_resource)
    {
        m_app->get_device()->CreateUnorderedAccessView(
            resource, counter_resource, &desc, get_cpu_handle(m_entries.size()));
        Descriptor_heap_handle handle(this, m_entries.size());
        Entry entry;

        m_entries.push_back(std::move(entry));
        return std::move(handle);
    }

    void Descriptor_heap::replace_by_unordered_access_view(
        const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc,
        ID3D12Resource* resource,
        ID3D12Resource* counter_resource,
        const Descriptor_heap_handle& handle)
    {
        m_app->get_device()->CreateUnorderedAccessView(
            resource, counter_resource, &desc, get_cpu_handle(handle));
    }

    // --------------------------------------------------------------------------------------------

    Descriptor_heap_handle Descriptor_heap::add_shader_resource_view(Texture* texture)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        if (!texture->get_srv_description(desc))
            return Descriptor_heap_handle();

        return add_shader_resource_view(desc, texture->get_resource());
    }

    Descriptor_heap_handle Descriptor_heap::add_unordered_access_view(Texture* texture)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
        if (!texture->get_uav_description(desc))
            return Descriptor_heap_handle();

        return add_unordered_access_view(desc, texture->get_resource(), nullptr);
    }

    bool Descriptor_heap::replace_by_unordered_access_view(
        Texture* texture, 
        const Descriptor_heap_handle& handle)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
        if (!texture->get_uav_description(desc))
            return false;

        replace_by_unordered_access_view(desc, texture->get_resource(), nullptr, handle);
        return true;
    }

    Descriptor_heap_handle Descriptor_heap::add_render_target_view(Texture* texture)
    {
        if (m_type != D3D12_DESCRIPTOR_HEAP_TYPE_RTV) {
            log_error("Render target views are supported by this type of heap: " + 
                      m_debug_name, SRC);
            return Descriptor_heap_handle();
        }

        m_app->get_device()->CreateRenderTargetView(
            texture->get_resource(), NULL, get_cpu_handle(m_entries.size()));

        Descriptor_heap_handle handle(this, m_entries.size());
        Entry entry;
        m_entries.push_back(std::move(entry));
        return std::move(handle);
    }


    bool Descriptor_heap::replace_by_render_target_view(
        Texture* texture, 
        const Descriptor_heap_handle& handle)
    {
        if (m_type != D3D12_DESCRIPTOR_HEAP_TYPE_RTV) {
            log_error("Render target views are supported by this type of heap: " + 
                      m_debug_name, SRC);
            return false;
        }

        m_app->get_device()->CreateRenderTargetView(
            texture->get_resource(), NULL, get_cpu_handle(handle));
        return true;
    }

    Descriptor_heap_handle Descriptor_heap::add_shader_resource_view(Buffer* buffer, bool raw)
    {
        if (!raw) {
            log_error("Only raw buffer views supported: " + m_debug_name, SRC);
            return Descriptor_heap_handle();
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        if (!buffer->get_shader_resource_view_description_raw(desc))
            return Descriptor_heap_handle();

        return add_shader_resource_view(desc, buffer->get_resource());
    }

    bool Descriptor_heap::replace_by_shader_resource_view(
        Buffer* buffer,
        bool raw, 
        const Descriptor_heap_handle& handle)
    {
        if (!raw)
        {
            log_error("Only raw buffer views supported: " + m_debug_name, SRC);
            return false;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        if (!buffer->get_shader_resource_view_description_raw(desc))
            return false;

        replace_by_shader_resource_view(desc, buffer->get_resource(), handle);
        return true;
    }
    

    Descriptor_heap_handle Descriptor_heap::add_shader_resource_view(
        Raytracing_acceleration_structure* tlas)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        if (!tlas->get_shader_resource_view_description(desc))
            return Descriptor_heap_handle();

        return add_shader_resource_view(desc, nullptr);
    }

    Descriptor_heap_handle Descriptor_heap::add_constant_buffer_view(
        const Constant_buffer_base* constants)
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc =
            constants->get_constant_buffer_view_description();
        return add_constant_buffer_view(desc);
    }

    bool Descriptor_heap::replace_by_constant_buffer_view(
        const Constant_buffer_base* constants, 
        const Descriptor_heap_handle& handle)
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc =
            constants->get_constant_buffer_view_description();

        replace_by_constant_buffer_view(desc, handle);
        return true;
    }
}
