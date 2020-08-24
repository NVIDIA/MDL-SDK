/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

 // examples/mdl_sdk/shared/gui.h

#ifndef EXAMPLE_SHARED_GUI_API_INTERFACE_DX12_H
#define EXAMPLE_SHARED_GUI_API_INTERFACE_DX12_H

#include "imgui.h"

#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include "gui/gui.h"

#include <d3d12.h>
#include <wrl.h>
#include <Windows.h>

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace mi { namespace examples { namespace gui
{
    class Api_interface_dx12 : public Api_interface
    {
    public:
        struct Render_context_dx12 : public Render_context
        {
            ID3D12GraphicsCommandList4* command_list;
        };

        explicit Api_interface_dx12(
            HWND window_handle,
            ID3D12Device5* device)
            : m_window_handle(window_handle)
            , m_device(device)
            , m_window_width(0)
            , m_window_height(0)
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc = {};
            desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = 1;
            desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_ui_heap));

            // Setup Platform / Renderer bindings
            ImGui_ImplWin32_Init(m_window_handle);
            ImGui_ImplDX12_Init(m_device, 2, DXGI_FORMAT_R8G8B8A8_UNORM,
                m_ui_heap.Get(),
                m_ui_heap->GetCPUDescriptorHandleForHeapStart(),
                m_ui_heap->GetGPUDescriptorHandleForHeapStart());
        }

        virtual ~Api_interface_dx12()
        {
            ImGui_ImplDX12_Shutdown();
            ImGui_ImplWin32_Shutdown();
        }

        void new_frame() override
        {
            RECT rect;
            ::GetClientRect(m_window_handle, &rect);
            m_window_width = size_t(rect.right) - size_t(rect.left);
            m_window_height = size_t(rect.bottom) - size_t(rect.top);

            ImGui_ImplDX12_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
        }

        void render(Render_context* context) override
        {
            Render_context_dx12* context_dx12 = static_cast<Render_context_dx12*>(context);
            ID3D12GraphicsCommandList4* command_list = context_dx12->command_list;

            std::vector<ID3D12DescriptorHeap*> heaps = { m_ui_heap.Get() };
            command_list->SetDescriptorHeaps(
                static_cast<UINT>(heaps.size()), heaps.data());

            ImGui::Render();
            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), command_list);
        }

        void get_window_size(size_t& out_width, size_t& out_height) const override
        {
            out_width = m_window_width;
            out_height = m_window_height;
        }

        /// call this function in the beginning of the window message handling and return true
        /// if the result of the call is true as the message was consumed by ImGui.
        /// Continue handling the window messages for you application otherwise.
        static bool handle_window_messages(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
        {
            return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
        }

    private:
        HWND m_window_handle;
        ID3D12Device5* m_device;

        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ui_heap;

        size_t m_window_width;
        size_t m_window_height;
    };
}}}
#endif
