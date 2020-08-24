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

// examples/mdl_sdk/dxr/mdl_d3d12/buffer.h

#ifndef MDL_D3D12_BUFFER_H
#define MDL_D3D12_BUFFER_H

#include "common.h"
#include "base_application.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;

    // --------------------------------------------------------------------------------------------

    class Buffer : public Resource
    {

    public:
        explicit Buffer(Base_application* app, size_t size_in_byte, std::string debug_name);
        virtual ~Buffer() = default;

        std::string get_debug_name() const override { return m_debug_name; }

        size_t get_size_in_byte() const { return m_size_in_byte; }

        template<typename T>
        bool set_data(const T* data, size_t element_count)
        {
            return set_data((void*) data, element_count * sizeof(T));
        }

        template<typename T>
        bool set_data(const std::vector<T>& data)
        {
            return set_data((void*) data.data(), data.size() * sizeof(T));
        }

        bool upload(D3DCommandList* command_list);

        ID3D12Resource* get_resource() const { return m_resource.Get(); }

        bool get_shader_resource_view_description_raw(D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const
        {
            desc = {};
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = static_cast<UINT>(m_size_in_byte / 4);
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            desc.Buffer.StructureByteStride = 0;
            return true;
        }

    protected:
        bool set_data(const void* data, size_t size_in_byte);

        Base_application* m_app;
        const std::string m_debug_name;
        const size_t m_size_in_byte;

    private:
        ComPtr<ID3D12Resource> m_resource;
        ComPtr<ID3D12Resource> m_upload_resource;
    };

    // --------------------------------------------------------------------------------------------

    template<typename TElement>
    class Structured_buffer : public Buffer
    {
    public:
        explicit Structured_buffer(
            Base_application* app, size_t element_count, std::string debug_name)
            : Buffer(app, element_count * sizeof(TElement), debug_name)
        {
        }
        virtual ~Structured_buffer() = default;

        bool get_shader_resource_view_description(D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const
        {
            desc = {};
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Buffer.NumElements = static_cast<UINT>(m_size_in_byte / sizeof(TElement));
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            desc.Buffer.StructureByteStride = static_cast<UINT>(sizeof(TElement));
            return true;
        }

        size_t get_element_count() const { return m_size_in_byte / sizeof(TElement); }
    };

    // --------------------------------------------------------------------------------------------

    template<typename TVertex>
    class Vertex_buffer : public Structured_buffer<TVertex>
    {
    public:
        explicit Vertex_buffer(Base_application* app, size_t element_count, std::string debug_name)
            : Structured_buffer<TVertex>(app, element_count, debug_name)
        {
        }
        virtual ~Vertex_buffer() = default;

        D3D12_VERTEX_BUFFER_VIEW get_vertex_buffer_view() const
        {
            D3D12_VERTEX_BUFFER_VIEW vertex_buffer_view;
            vertex_buffer_view.BufferLocation = Buffer::get_resource()->GetGPUVirtualAddress();
            vertex_buffer_view.StrideInBytes = static_cast<uint32_t>(sizeof(TVertex));
            vertex_buffer_view.SizeInBytes = static_cast<uint32_t>(Buffer::get_size_in_byte());
            return std::move(vertex_buffer_view);
        }
    };

    // --------------------------------------------------------------------------------------------

    class Index_buffer : public Buffer
    {
    public:
        explicit Index_buffer(Base_application* app, size_t element_count, std::string debug_name)
            : Buffer(app, element_count * sizeof(uint32_t), debug_name)
        {
        }

        D3D12_INDEX_BUFFER_VIEW get_index_buffer_view() const
        {
            D3D12_INDEX_BUFFER_VIEW index_buffer_view;
            index_buffer_view.BufferLocation = get_resource()->GetGPUVirtualAddress();
            index_buffer_view.Format = DXGI_FORMAT_R32_UINT;
            index_buffer_view.SizeInBytes = static_cast<uint32_t>(Buffer::get_size_in_byte());
            return index_buffer_view;
        }

        bool get_shader_resource_view_description(D3D12_SHADER_RESOURCE_VIEW_DESC& desc) const
        {
            desc = {};
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Buffer.NumElements = static_cast<UINT>(m_size_in_byte / sizeof(uint32_t));
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            desc.Buffer.StructureByteStride = static_cast<UINT>(sizeof(uint32_t));
            return true;
        }

        size_t get_element_count() const { return m_size_in_byte / sizeof(uint32_t); }
    };

    // --------------------------------------------------------------------------------------------

    class Constant_buffer_base : public Resource
    {
    public:
        explicit Constant_buffer_base(
            Base_application* app,
            size_t size_in_byte,
            std::string debug_name)

            : m_app(app)
            , m_mapped_data(nullptr)
            , m_debug_name(debug_name)
            , m_size_in_byte(
                round_to_power_of_two(size_in_byte,
                D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT))
        {
            if (log_on_failure(m_app->get_device()->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(m_size_in_byte),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&m_resource)),
                "Failed to create resource of " + std::to_string(m_size_in_byte) + " bytes for: " + m_debug_name, SRC))
                return;

            set_debug_name(m_resource.Get(), m_debug_name);

            // map buffer and keep it mapped
            if (log_on_failure(
                m_resource->Map(0, nullptr, reinterpret_cast<void**>(&m_mapped_data)),
                "Failed to map buffer: " + m_debug_name, SRC))
                return;
            memset(m_mapped_data, 0, m_size_in_byte);
        }

        virtual ~Constant_buffer_base() = default;

        std::string get_debug_name() const override { return m_debug_name; }

        ID3D12Resource* get_resource() const { return m_resource.Get(); }

        bool get_constant_buffer_view_description(D3D12_CONSTANT_BUFFER_VIEW_DESC& desc) const
        {
            desc.BufferLocation = m_resource->GetGPUVirtualAddress();
            desc.SizeInBytes = static_cast<UINT>(m_size_in_byte);
            return true;
        }

        bool get_shader_resource_view_description(
            D3D12_SHADER_RESOURCE_VIEW_DESC& desc, bool raw) const
        {
            if (!raw)
            {
                log_error("Only raw buffer views are implemented yet: " + m_debug_name, SRC);
                return false;
            }

            desc = {};
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = static_cast<UINT>(m_size_in_byte / 4);
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            desc.Buffer.StructureByteStride = 0;
            return true;
        }

        // upload constant data to the GPU
        virtual void upload() = 0;

    protected:
        Base_application* m_app;
        void* m_mapped_data;
        const std::string m_debug_name;

    private:
        size_t m_size_in_byte;
        ComPtr<ID3D12Resource> m_resource;
    };

    // --------------------------------------------------------------------------------------------

    template<typename T> class Dynamic_constant_buffer;

    template<typename TConstantStruct>
    class Constant_buffer : public Constant_buffer_base
    {
        friend Dynamic_constant_buffer<TConstantStruct>;
    public:
        explicit Constant_buffer(Base_application* app, std::string debug_name)
            : Constant_buffer_base(app, sizeof(TConstantStruct), debug_name)
        {
            memset(&data, 0, sizeof(TConstantStruct));
        }

        virtual ~Constant_buffer() = default;

        /// constant data to be copied on update
        TConstantStruct data;

        // upload constant data to the GPU
        void upload() override
        {
            memcpy(m_mapped_data, &data, sizeof(TConstantStruct));
        }
    };

    // --------------------------------------------------------------------------------------------

    template<typename TConstantStruct>
    class Dynamic_constant_buffer
    {
    public:
        explicit Dynamic_constant_buffer(
            Base_application* app,
            std::string debug_name,
            size_t frame_buffer_count)
            : m_buffers(0)
            , m_next_frame_index(0)
        {
            m_buffers.reserve(frame_buffer_count);
            for (size_t i = 0; i < frame_buffer_count; ++i)
            {
                m_buffers.push_back(new Constant_buffer<TConstantStruct>(
                    app, debug_name + "_" + std::to_string(i)));
            }
            memset(&m_data, 0, sizeof(TConstantStruct));
        }

        virtual ~Dynamic_constant_buffer()
        {
            for (auto b : m_buffers)
                delete b;

            m_buffers.clear();
        }

        /// constant data to be copied on update
        TConstantStruct& data() { return m_data; }
        const TConstantStruct& data() const { return m_data; }

        D3D12_GPU_VIRTUAL_ADDRESS bind(const Render_args& args) const
        {
            memcpy(m_buffers[m_next_frame_index]->m_mapped_data, &m_data, sizeof(TConstantStruct));

            D3D12_GPU_VIRTUAL_ADDRESS address =
                m_buffers[m_next_frame_index]->get_resource()->GetGPUVirtualAddress();
            m_next_frame_index = args.frame_number % 2;
            return address;
        }

    private:
        TConstantStruct m_data;
        std::vector<Constant_buffer<TConstantStruct>*> m_buffers;
        mutable size_t m_next_frame_index;
    };

}}} // mi::examples::mdl_d3d12
#endif
