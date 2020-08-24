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

#include "window_win32.h"
#include "texture.h"
#include "base_application.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include <gui/gui.h>
#include <gui/gui_api_interface_dx12.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    Window_win32* window = reinterpret_cast<Window_win32*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
    Base_application_message_interface* app = nullptr;

    if (window && mi::examples::gui::Api_interface_dx12::handle_window_messages(
        hWnd, message, wParam, lParam))
            return true;

    if(window)
        app = &window->m_message_pump_interface;

    if (window && app)
    {
        for (auto&& callback : window->m_message_callbacks)
        {
            if (LRESULT res = callback(hWnd, message, wParam, lParam))
                return res;
        }
    }

    switch (message)
    {
        case WM_CREATE:
        {
            LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(
                pCreateStruct->lpCreateParams));
        }
        return 0;

        case WM_KEYDOWN:
            if (app)
                app->key_down(static_cast<uint8_t>(wParam));
            return 0;

        case WM_KEYUP:
            if (app)
                app->key_up(static_cast<uint8_t>(wParam));
            return 0;

        case WM_SIZE:
            if (app)
            {
                if (wParam == SIZE_MINIMIZED) // window minimized
                    return 0;

                RECT clientRect = {};
                GetClientRect(hWnd, &clientRect);
                size_t width = clientRect.right - clientRect.left;
                size_t height = clientRect.bottom - clientRect.top;

                app->resize(width, height, 96.0);
            }
            return 0;

        case WM_DISPLAYCHANGE:
            return 0;

        case WM_MOVE:
            return 0;

        case WM_PAINT:
            if (app)
                app->paint();
            return 0;

        case WM_GETMINMAXINFO:
        {
            LPMINMAXINFO lpMMI = (LPMINMAXINFO) lParam;
            lpMMI->ptMinTrackSize.x = 320;
            lpMMI->ptMinTrackSize.y = 240;
            return 0;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
}

// ------------------------------------------------------------------------------------------------

Window_win32::Window_win32(Base_application_message_interface& message_pump_interface)
    : m_app(message_pump_interface.get_application())
    , m_message_pump_interface(message_pump_interface)
    , m_window_handle(0)
    , m_vsync(true)
    , m_mode(IWindow::Mode::Windowed)
    , m_close(false)
    , m_gui(nullptr)
{
    auto options = m_app->get_options();

    WNDCLASSEX windowClass = {0};
    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = m_message_pump_interface.get_win32_instance_handle();
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.lpszClassName = L"MDL_d3d12_class";
    RegisterClassEx(&windowClass);

    RECT windowRect = {
        0,
        0,
        static_cast<LONG>(options->window_width),
        static_cast<LONG>(options->window_height)
    };
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    m_window_handle = CreateWindow(
        windowClass.lpszClassName,
        options->window_title.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        windowRect.right - windowRect.left,
        windowRect.bottom - windowRect.top,
        nullptr,
        nullptr,
        m_message_pump_interface.get_win32_instance_handle(),
        this);

    // get render area size
    RECT rect;
    bool success = GetClientRect(m_window_handle, &rect) != 0;
    if(!success)
        log_error("Failed to get current window size.", SRC);

    m_width = success ? (rect.right - rect.left) : 320;
    m_height = success ? (rect.bottom - rect.top) : 240;
    m_dpi = 96.0f;

    // create swap chain
    DXGI_SWAP_CHAIN_DESC1 desc = {};
    desc.BufferCount = 2;
    desc.Width = static_cast<UINT>(m_width);
    desc.Height = static_cast<UINT>(m_height);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.SampleDesc.Count = 1;
    //desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    ComPtr<IDXGISwapChain1> swapChain;
    throw_on_failure(m_app->get_dxgi_factory()->CreateSwapChainForHwnd(
        m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT)->get_queue(),
        m_window_handle,
        &desc,
        nullptr,
        nullptr,
        &swapChain),
        "Failed to create Swap Chain.", SRC);

    throw_on_failure(m_app->get_dxgi_factory()->MakeWindowAssociation(
        m_window_handle, DXGI_MWA_NO_ALT_ENTER),
        "Failed to set window flags.", SRC);

    throw_on_failure(swapChain.As(&m_swap_chain),
        "Failed to cast swap chain interface.", SRC);

    // create back buffers
    m_swap_buffer_index = m_swap_chain->GetCurrentBackBufferIndex();

    // create views on the heap
    Descriptor_heap* target_heap = m_app->get_render_target_descriptor_heap();
    Descriptor_heap_handle first_handle = target_heap->reserve_views(desc.BufferCount);

    for (size_t i = 0; i < desc.BufferCount; ++i)
    {
        Texture* target = new Texture(
            m_app, m_swap_chain.Get(), i, "BackBuffer" + std::to_string(i));
        m_swap_buffers.push_back(target);
        m_swap_fence_handles.push_back(
            m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT)->get_fence()->signal());

        Descriptor_heap_handle handle = first_handle.create_offset(i);
        if (!target_heap->create_render_target_view(target, handle)) {
            std::string msg = "Failed to create resource view for back buffer.";
            log_error(msg, SRC);
            throw(msg);
        }

        m_render_target_views_heap_indices.push_back(handle);
    }

    // create platform specific UI boilerplate
    auto api_interface = std::make_unique<mi::examples::gui::Api_interface_dx12>(
        m_window_handle, m_app->get_device());

    // create the main GUI instance for the application window referenced by the API interface.
    // and transfer the interface to the main GUI instance
    m_gui = new mi::examples::gui::Root(std::move(api_interface));
}

// ------------------------------------------------------------------------------------------------

Window_win32::~Window_win32()
{
    if (m_gui)
        delete m_gui;

    for (auto&& t : m_swap_buffers)
        delete t;
}

// ------------------------------------------------------------------------------------------------

int Window_win32::show(int nCmdShow)
{
    ShowWindow(m_window_handle, nCmdShow);

    MSG msg = {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (m_close)
            return 0;
    }

    return static_cast<int>(msg.wParam);
}

// ------------------------------------------------------------------------------------------------

void Window_win32::close()
{
    set_window_mode(IWindow::Mode::Windowed);
    m_close = true;
}

// ------------------------------------------------------------------------------------------------

bool Window_win32::has_focus() const
{
    return GetForegroundWindow() == m_window_handle;
}

// ------------------------------------------------------------------------------------------------

Texture* Window_win32::get_back_buffer() const
{
    return m_swap_buffers[m_swap_buffer_index];
}

// ------------------------------------------------------------------------------------------------

D3D12_CPU_DESCRIPTOR_HANDLE Window_win32::get_back_buffer_rtv() const
{
    return m_render_target_views_heap_indices[m_swap_buffer_index].get_cpu_handle();
}

// ------------------------------------------------------------------------------------------------

bool Window_win32::present_back_buffer()
{
    if (FAILED(m_swap_chain->Present(m_vsync ? 1 : 0, 0)))
    {
        log_error("Failed to present back buffer.", SRC);
        return false;
    }

    auto fence = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT)->get_fence();
    m_swap_fence_handles[m_swap_buffer_index] = fence->signal();
    m_swap_buffer_index = m_swap_chain->GetCurrentBackBufferIndex();
    fence->wait(m_swap_fence_handles[m_swap_buffer_index]);
    return true;
}

// ------------------------------------------------------------------------------------------------

void Window_win32::set_window_mode(IWindow::Mode mode)
{
    BOOL fullscreen_state;
    throw_on_failure(m_swap_chain->GetFullscreenState(&fullscreen_state, nullptr),
        "Failed to get current window mode.", SRC);

    if (mode == IWindow::Mode::Windowed && !fullscreen_state) return;
    if (mode == IWindow::Mode::Fullsceen && fullscreen_state) return;

    // make sure all current work is done
    m_app->flush_command_queues();

    // actually set the new mode
    if (log_on_failure(m_swap_chain->SetFullscreenState(!fullscreen_state, nullptr),
        "Failed to change window mode.", SRC))
        return;

    m_mode = mode;
}

// ------------------------------------------------------------------------------------------------

bool Window_win32::resize(size_t width, size_t height, double dpi)
{
    if (width == m_width && height == m_height && dpi == m_dpi)
        return true;

    m_width = width;
    m_height = height;

    for (auto&& buffer : m_swap_buffers)
        delete buffer;

    DXGI_SWAP_CHAIN_DESC desc;
    throw_on_failure(m_swap_chain->GetDesc(&desc),
        "Failed to get Swap Chain description.", SRC);

    throw_on_failure(m_swap_chain->ResizeBuffers(
        desc.BufferCount,
        static_cast<UINT>(width),
        static_cast<UINT>(height),
        desc.BufferDesc.Format,
        desc.Flags),
        "Failed to resize Swap Chain.", SRC);

    for (size_t i = 0; i < desc.BufferCount; ++i)
    {
        m_swap_buffers[i] =
            new Texture(m_app, m_swap_chain.Get(), i, "BackBuffer" + std::to_string(i));

        m_swap_fence_handles[i] =
            m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT)->get_fence()->signal();

        if (!m_app->get_render_target_descriptor_heap()->create_render_target_view(
            m_swap_buffers[i], m_render_target_views_heap_indices[i]))
            return false;
    }
    m_swap_buffer_index = m_swap_chain->GetCurrentBackBufferIndex();
    return true;
}

// ------------------------------------------------------------------------------------------------

void Window_win32::add_message_callback(
    std::function<LRESULT(HWND, UINT, WPARAM, LPARAM)> callback)
{
    m_message_callbacks.push_back(callback);
}

}}} // mi::examples::mdl_d3d12
