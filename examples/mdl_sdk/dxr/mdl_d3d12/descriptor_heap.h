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

// examples/mdl_sdk/dxr/mdl_d3d12/descriptor_heap.h

#ifndef MDL_D3D12_DESCRIPTOR_HEAP_H
#define MDL_D3D12_DESCRIPTOR_HEAP_H

#include "common.h"
#include "base_application.h"
#include "buffer.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Raytracing_acceleration_structure;
    class Texture;
    enum class Texture_dimension;

    // --------------------------------------------------------------------------------------------

    class Descriptor_heap
    {
        // TODO supposed to be used to track states and improve error handling
        struct Entry
        {
            enum class Kind
            {
                Unknown = 0,
                SRV,    // shader resource view
                CBV,    // constant buffer view
                RTV,    // render target view
                UAV,    // unordered access view
            };

            explicit Entry();
            virtual ~Entry() = default;

            friend class Descriptor_heap;
        private:
            std::string resource_name;
            Kind resource_type;
            size_t alloc_block_id;
            size_t alloc_block_size;
            size_t alloc_block_index;
        };

    public:
        explicit Descriptor_heap(Base_application* app,
                                 D3D12_DESCRIPTOR_HEAP_TYPE type,
                                 size_t size,
                                 std::string debug_name);
        virtual ~Descriptor_heap();

        /// Reserves a number of resource views on the heap and returns a handle to the first one.
        /// Used to create handles that then be used with the `create_*_view(...)` methods.
        Descriptor_heap_handle reserve_views(size_t count);

        /// Allows the block that was reserved with the given handle to be reused later.
        /// Note, all the handles of the block must not be used anymore.
        void free_views(Descriptor_heap_handle& handle);

        /// Returns the size of the block a given handle belongs to.
        /// The block size corresponds to the 'count' that was passed to 'reserve_views'.
        size_t get_block_size(const Descriptor_heap_handle& handle);

        /// Create a Shader Resource View (SRV) at a given position on the heap.
        bool create_shader_resource_view(
            Buffer* buffer, bool raw, const Descriptor_heap_handle& handle);

        /// Create a Shader Resource View (SRV) at a given position on the heap.
        bool create_shader_resource_view(
            Texture* texture,
            Texture_dimension dimension,
            const Descriptor_heap_handle& handle);

        /// Create a Shader Resource View (SRV) at a given position on the heap.
        bool create_shader_resource_view(
            Raytracing_acceleration_structure* tlas, const Descriptor_heap_handle& handle);

        /// Create a Shader Resource View (SRV) at a given position on the heap.
        template<typename T> bool create_shader_resource_view(
            Structured_buffer<T>* buffer,
            const Descriptor_heap_handle& handle)
        {
            if (!handle.is_valid()) {
                log_error("Heap Handle invalid while creating view to: " +
                    buffer->get_debug_name(), SRC);
                return false;
            }

            D3D12_SHADER_RESOURCE_VIEW_DESC desc;
            if (!buffer->get_shader_resource_view_description(desc))
                return false;

            std::lock_guard<std::mutex> lock(m_entries_mutex);
            m_entries[handle].resource_name = buffer->get_debug_name();
            m_entries[handle].resource_type = Entry::Kind::SRV;
            m_app->get_device()->CreateShaderResourceView(
                buffer->get_resource(), &desc, handle.get_cpu_handle());
            return true;
        }

        /// Create a Render Target View (RTV) at a given position on the heap.
        bool create_render_target_view(
            Texture* texture, const Descriptor_heap_handle& handle);

        /// Create an Unordered Access View (UAV) at a given position on the heap.
        bool create_unordered_access_view(
            Texture* texture, const Descriptor_heap_handle& handle);

        /// Create a Constant Buffer View (CBV) at a given position on the heap.
        bool create_constant_buffer_view(
            const Constant_buffer_base* constants, const Descriptor_heap_handle& handle);

        /// Get the internal D3D heap
        ID3D12DescriptorHeap* get_heap() { return m_heap.Get(); }

        /// Print the heap structure to the log
        void print_debug_infos();

    private:
        friend D3D12_CPU_DESCRIPTOR_HANDLE Descriptor_heap_handle::get_cpu_handle() const;
        friend D3D12_GPU_DESCRIPTOR_HANDLE Descriptor_heap_handle::get_gpu_handle() const;

        Base_application* m_app;
        const std::string m_debug_name;
        D3D12_DESCRIPTOR_HEAP_TYPE m_type;
        size_t m_size;
        size_t m_element_size;

        std::mutex m_entries_mutex;
        std::vector<Entry> m_entries;
        std::atomic<size_t> m_entry_alloc_block_counter;

        std::map<size_t, std::stack<size_t>> m_unused_entries_by_size;

        ComPtr<ID3D12DescriptorHeap> m_heap;
        D3D12_CPU_DESCRIPTOR_HANDLE m_cpu_heap_start;

        D3D12_GPU_DESCRIPTOR_HANDLE m_gpu_heap_start;
    };

}}} // mi::examples::mdl_d3d12
#endif
