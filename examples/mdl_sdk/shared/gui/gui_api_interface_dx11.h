/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

 // examples/mdl_sdk/shared/gui.h

#ifndef EXAMPLE_SHARED_GUI_API_INTERFACE_DX11_H
#define EXAMPLE_SHARED_GUI_API_INTERFACE_DX11_H

#include "imgui.h"

#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "gui/gui.h"
#include <d3d11.h>

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace mi { namespace examples { namespace gui
{
    class Api_interface_dx11 : public Api_interface
    {
    public:
        explicit Api_interface_dx11(
            HWND window_handle,
            ID3D11Device* device,
            ID3D11DeviceContext* device_context)
            : m_window_handle(window_handle)
            , m_device(device)
            , m_device_context(device_context)
            , m_window_width(0)
            , m_window_height(0)
        {
            // Setup Platform / Renderer bindings
            ImGui_ImplWin32_Init(m_window_handle);
            ImGui_ImplDX11_Init(m_device, m_device_context);
        }

        virtual ~Api_interface_dx11()
        {
            ImGui_ImplDX11_Shutdown();
            ImGui_ImplWin32_Shutdown();
        }

        void new_frame() override
        {
            RECT rect;
            ::GetClientRect(m_window_handle, &rect);
            m_window_width = size_t(rect.right) - size_t(rect.left);
            m_window_height = size_t(rect.bottom) - size_t(rect.top);

            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
        }

        void render(Render_context* /*context*/) override
        {
            ImGui::Render();
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
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
        ID3D11Device* m_device;
        ID3D11DeviceContext* m_device_context;

        size_t m_window_width;
        size_t m_window_height;
    };
}}}
#endif
