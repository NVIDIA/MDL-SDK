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

#include "descriptor_heap.h"
#include "raytracing_pipeline.h"
#include "texture.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Descriptor_heap_handle::Descriptor_heap_handle()
    : m_descriptor_heap(nullptr)
    , m_index(static_cast<size_t>(-1))
{
}

// ------------------------------------------------------------------------------------------------

Descriptor_heap_handle::Descriptor_heap_handle(Descriptor_heap* heap, size_t index)
    : m_descriptor_heap(heap)
    , m_index(index)
{
}

// ------------------------------------------------------------------------------------------------

Descriptor_heap_handle Descriptor_heap_handle::create_offset(size_t offset)
{
    return Descriptor_heap_handle(m_descriptor_heap, m_index + offset);
}

// ------------------------------------------------------------------------------------------------

D3D12_CPU_DESCRIPTOR_HANDLE Descriptor_heap_handle::get_cpu_handle() const
{
    if (!m_descriptor_heap)
        return D3D12_CPU_DESCRIPTOR_HANDLE{ NULL };

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(m_descriptor_heap->m_cpu_heap_start);
    cpu_handle.ptr += m_index * m_descriptor_heap->m_element_size;
    return cpu_handle;
}

// ------------------------------------------------------------------------------------------------

D3D12_GPU_DESCRIPTOR_HANDLE Descriptor_heap_handle::get_gpu_handle() const
{
    if (!m_descriptor_heap)
        return D3D12_GPU_DESCRIPTOR_HANDLE{ NULL };

    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(m_descriptor_heap->m_gpu_heap_start);
    gpu_handle.ptr += m_index * m_descriptor_heap->m_element_size;
    return gpu_handle;
}

// ------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------

Descriptor_heap::Entry::Entry()
    : resource_name("")
    , resource_type(Entry::Kind::Unknown)
    , alloc_block_id(size_t(-1))
    , alloc_block_size(1)
    , alloc_block_index(0)
{
}

// --------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

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
    , m_entries()
    , m_entries_mutex()
    , m_entry_alloc_block_counter(0)
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

// ------------------------------------------------------------------------------------------------

Descriptor_heap::~Descriptor_heap()
{
}

// ------------------------------------------------------------------------------------------------

Descriptor_heap_handle Descriptor_heap::reserve_views(size_t count)
{
    if (count == 0)
        return Descriptor_heap_handle();

    Descriptor_heap_handle first_handle;
    std::lock_guard<std::mutex> lock(m_entries_mutex);

    // try to reuse a free entry of this size or larger
    auto it = m_unused_entries_by_size.lower_bound(count);
    if (it != m_unused_entries_by_size.end() && !it->second.empty())
    {
        size_t index = it->second.top();
        it->second.pop();
        Entry first_entry = m_entries[index];
        return Descriptor_heap_handle(this, index);
    }

    if (m_entries.size() + count > m_size)
    {
        // TODO resize
        log_error("Number of resources views to reserve exceeds the size of the heap.", SRC);
    }
    else
    {
        Entry e;
        e.alloc_block_id = m_entry_alloc_block_counter.fetch_add(1);
        e.alloc_block_size = count;
        first_handle = Descriptor_heap_handle(this, m_entries.size());
        for (size_t i = 0; i < count; ++i)
        {
            e.alloc_block_index = i;
            m_entries.push_back(e);
        }
    }
    return std::move(first_handle);
}

// ------------------------------------------------------------------------------------------------

void Descriptor_heap::free_views(Descriptor_heap_handle& handle)
{
    std::lock_guard<std::mutex> lock(m_entries_mutex);

    Entry entry = m_entries[handle];
    size_t index = handle - entry.alloc_block_index;
    for (size_t i = 0; i < entry.alloc_block_size; ++i)
    {
        m_entries[index + i].resource_name = "(unused)";
        m_entries[index + i].resource_type = Entry::Kind::Unknown;
    }
    m_unused_entries_by_size[entry.alloc_block_size].push(index);
    handle = Descriptor_heap_handle(); // invalid
}

// ------------------------------------------------------------------------------------------------

size_t Descriptor_heap::get_block_size(const Descriptor_heap_handle& handle)
{
    std::lock_guard<std::mutex> lock(m_entries_mutex);

    if (!handle.is_valid())
    {
        log_error("Heap Handle invalid while getting block size.", SRC);
        return 0;
    }

    return m_entries[handle].alloc_block_size;
}

// ------------------------------------------------------------------------------------------------

void Descriptor_heap::print_debug_infos()
{
    std::lock_guard<std::mutex> lock(m_entries_mutex);
    std::string msg = "Heap Information: " + m_debug_name;
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        const Entry& e = m_entries[i];

        msg += "\n  [" + std::to_string(i) + "]\t " +
            "[block " + std::to_string(e.alloc_block_id) +
            " (" + std::to_string(e.alloc_block_index + 1) + "/" +
            std::to_string(e.alloc_block_size) + ")] [";
        switch (e.resource_type)
        {
        case Entry::Kind::SRV: msg += "SRV"; break;
        case Entry::Kind::CBV: msg += "CBV"; break;
        case Entry::Kind::RTV: msg += "RTV"; break;
        case Entry::Kind::UAV: msg += "UAV"; break;
        default: msg += "Unknown"; break;
        }

        msg += "] " + e.resource_name;
    }
    log_info(msg);
}

// --------------------------------------------------------------------------------------------

bool Descriptor_heap::create_shader_resource_view(
    Texture* texture,
    Texture_dimension dimension,
    const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            (texture ? texture->get_debug_name() : "NullView"), SRC);
        return false;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC desc {};
    if (texture)
    {
        if (!texture->get_srv_description(desc, dimension))
            return false;
    }
    else
    {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        switch (dimension)
        {
            case Texture_dimension::Texture_2D:
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                break;
            case Texture_dimension::Texture_3D:
                desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                break;
            default:
                log_error("Texture has no valid dimension: NullView", SRC);
                return false;
        }
    }

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = texture ? texture->get_debug_name() : "NullView";
    m_entries[handle].resource_type = Entry::Kind::SRV;
    m_app->get_device()->CreateShaderResourceView(
        texture ? texture->get_resource() : nullptr,
        &desc,
        handle.get_cpu_handle());
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Descriptor_heap::create_unordered_access_view(
    Texture* texture,
    const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            texture->get_debug_name(), SRC);
        return false;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
    if (!texture->get_uav_description(desc))
        return false;

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = texture->get_debug_name();
    m_entries[handle].resource_type = Entry::Kind::UAV;
    m_app->get_device()->CreateUnorderedAccessView(
        texture->get_resource(), nullptr, &desc, handle.get_cpu_handle());
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Descriptor_heap::create_render_target_view(
    Texture* texture,
    const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            texture->get_debug_name(), SRC);
        return false;
    }

    if (m_type != D3D12_DESCRIPTOR_HEAP_TYPE_RTV) {
        log_error("Render target views are supported by this type of heap: " +
            m_debug_name, SRC);
        return false;
    }

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = texture->get_debug_name();
    m_entries[handle].resource_type = Entry::Kind::RTV;
    m_app->get_device()->CreateRenderTargetView(
        texture->get_resource(), NULL, handle.get_cpu_handle());
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Descriptor_heap::create_shader_resource_view(
    Buffer* buffer,
    bool raw,
    const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            buffer->get_debug_name(), SRC);
        return false;
    }

    if (!raw) {
        log_error("Only raw buffer views supported: " + m_debug_name, SRC);
        return false;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC desc;
    if (!buffer->get_shader_resource_view_description_raw(desc))
        return false;

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = buffer->get_debug_name();
    m_entries[handle].resource_type = Entry::Kind::SRV;
    m_app->get_device()->CreateShaderResourceView(
        buffer->get_resource(), &desc, handle.get_cpu_handle());
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Descriptor_heap::create_shader_resource_view(
    Raytracing_acceleration_structure* tlas, const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            tlas->get_debug_name(), SRC);
        return false;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC desc;
    if (!tlas->get_shader_resource_view_description(desc))
        return false;

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = tlas->get_debug_name();
    m_entries[handle].resource_type = Entry::Kind::SRV;
    m_app->get_device()->CreateShaderResourceView(
        nullptr, &desc, handle.get_cpu_handle());
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Descriptor_heap::create_constant_buffer_view(
    const Constant_buffer_base* constants,
    const Descriptor_heap_handle& handle)
{
    if (!handle.is_valid()) {
        log_error("Heap Handle invalid while creating view to: " +
            constants->get_debug_name(), SRC);
        return false;
    }

    D3D12_CONSTANT_BUFFER_VIEW_DESC desc;
    if (!constants->get_constant_buffer_view_description(desc))
        return false;

    std::lock_guard<std::mutex> lock(m_entries_mutex);
    m_entries[handle].resource_name = constants->get_debug_name();
    m_entries[handle].resource_type = Entry::Kind::CBV;
    m_app->get_device()->CreateConstantBufferView(
        &desc, handle.get_cpu_handle());
    return true;
}

}}} // mi::examples::mdl_d3d12
