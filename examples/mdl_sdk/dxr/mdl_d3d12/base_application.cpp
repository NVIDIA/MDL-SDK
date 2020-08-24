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

#include "base_application.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include "dxgidebug.h"
#include "mdl_sdk.h"
#include "window.h"
#include "window_image_file.h"
#include "window_win32.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Base_application_message_interface::Base_application_message_interface(
    Base_application* app,
    HINSTANCE instance)

    : m_app(app)
    , m_instance(instance)
{
}

// ------------------------------------------------------------------------------------------------

void Base_application_message_interface::key_down(uint8_t key)
{
    m_app->key_down(key);
}

void Base_application_message_interface::key_up(uint8_t key)
{
    m_app->key_up(key);
}

// ------------------------------------------------------------------------------------------------

void Base_application_message_interface::paint()
{
    m_app->update();
    m_app->render();
}

// ------------------------------------------------------------------------------------------------

void Base_application_message_interface::resize(size_t width, size_t height, double dpi)
{
    m_app->flush_command_queues();
    m_app->m_window->resize(width, height, dpi);
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Base_application::Base_application()
    : m_window(nullptr)
    , m_scene_is_updating_next(false)
{
    m_update_args.frame_number = 0;
    m_update_args.elapsed_time = 0.0;
    m_update_args.total_time = 0.0;

    m_render_args.frame_number = 0;
    m_render_args.back_buffer = nullptr;
    m_render_args.back_buffer_rtv = {0};
}

// ------------------------------------------------------------------------------------------------

Base_application::~Base_application()
{
    log_set_file_path(nullptr); // close the file if there is one
}

// ------------------------------------------------------------------------------------------------

int Base_application::run(Base_options* options, HINSTANCE hInstance, int nCmdShow)
{
    // create graphics context, load MDL SDK, ...
    if (!initialize_internal(options)) return -1;

    // create the window
    Base_application_message_interface message_interface(this, hInstance);
    if (options->no_gui)
        m_window = new Window_image_file(
            message_interface, m_options->output_file, m_options->iterations);
    else
        m_window = new Window_win32(message_interface);

    m_render_args.backbuffer_width = m_window->get_width();
    m_render_args.backbuffer_height = m_window->get_height();

    int return_code = 0;
    // load the applications content and then run the main loop
    if (load())
    {
        // show the window and run the message loop
        int return_code = m_window->show(nCmdShow);
        if (return_code != 0)
            log_warning("Applications main loop stopped with issues.", SRC);

        // complete the current work load
        flush_command_queues();
    }
    else
        log_error("Loading Applications failed. Freeing already loaded content.", SRC);

    // unload the application
    if (!unload())
        log_error("Unloading Applications failed.", SRC);

    // release base application resources
    for (auto&& queue : m_command_queues)
        delete queue.second;

    delete m_window;
    delete m_mdl_sdk;
    delete m_resource_descriptor_heap;
    delete m_render_target_descriptor_heap;
    m_device = nullptr;
    m_factory = nullptr;

    #if defined(_DEBUG)
    {
        ComPtr<IDXGIDebug1> debugController;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&debugController))))
        {
            debugController->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_ALL);
        }
    }
    #endif

    return return_code;
}

// ------------------------------------------------------------------------------------------------

Command_queue* Base_application::get_command_queue(D3D12_COMMAND_LIST_TYPE type)
{
    auto found = m_command_queues.find(type);
    if (found != m_command_queues.end())
        return found->second;

    Command_queue* new_queue = new Command_queue(this, type);
    m_command_queues[type] = new_queue;
    return new_queue;
}

// ------------------------------------------------------------------------------------------------

void Base_application::flush_command_queues()
{
    for (auto&& it : m_command_queues)
        it.second->flush();
}

// ------------------------------------------------------------------------------------------------

Descriptor_heap* Base_application::get_resource_descriptor_heap()
{
    { return m_resource_descriptor_heap; }
}

// ------------------------------------------------------------------------------------------------

Descriptor_heap* Base_application::get_render_target_descriptor_heap()
{
    { return m_render_target_descriptor_heap; }
}

// ------------------------------------------------------------------------------------------------

bool Base_application::initialize_internal(Base_options* options)
{
    if (!initialize(options))
        return false;

    m_options = options;

    UINT dxgi_factory_flags = 0;
    D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_12_0;

    #if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
        {
            debugController->EnableDebugLayer();
        }

        ComPtr<IDXGIInfoQueue> dxgiInfoQueue;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiInfoQueue))))
        {
            dxgi_factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
            dxgiInfoQueue->SetBreakOnSeverity(
                DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);
            dxgiInfoQueue->SetBreakOnSeverity(
                DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, false);
            dxgiInfoQueue->SetBreakOnSeverity(
                DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_WARNING, false);
            dxgiInfoQueue->SetBreakOnSeverity(
                DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_INFO, false);
            dxgiInfoQueue->SetBreakOnSeverity(
                DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_MESSAGE, false);
        }
    }
    #endif

    if (log_on_failure(CreateDXGIFactory2(dxgi_factory_flags, IID_PPV_ARGS(&m_factory)),
        "Failed to create DXGI Factory.", SRC))
        return false;


    // collect non-software adapters
    bool found_adapter = false;
    struct adapter_pair
    {
        ComPtr<IDXGIAdapter1> adapter;
        DXGI_ADAPTER_DESC1 desc;
    };

    std::deque<adapter_pair> available_adapters;

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT a = 0; DXGI_ERROR_NOT_FOUND != m_factory->EnumAdapters1(a, &adapter); ++a)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // check if a device fits the requirements
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        available_adapters.push_back({adapter, desc});
    }

    // sort by dedicated memory, assuming that is a good heuristic
    std::sort(available_adapters.begin(), available_adapters.end(),
        [](const adapter_pair& a, const adapter_pair& b)
        {
            return a.desc.DedicatedVideoMemory > b.desc.DedicatedVideoMemory;
        });

    // allow the user to select a certain GPU
    if (available_adapters.size() > 1)
    {
        std::string msg =
            "Multiple GPUs detected, run with option '--gpu <num>' to select a specific one."
            "\n                      Default is the first one (from the top) that supports RTX:";

        for (size_t i = 0; i < available_adapters.size(); ++i)
        {
            msg += "\n                      - [" + std::to_string(i) + "] " +
                mi::examples::strings::wstr_to_str(available_adapters[i].desc.Description);
        }

        log_info(msg);
    }

    // if the user picked one, move that to the top
    if (options->gpu >= 0 && options->gpu < available_adapters.size())
    {
        adapter_pair selected = available_adapters[options->gpu];
        available_adapters.erase(available_adapters.begin() + options->gpu);
        available_adapters.push_front(selected);
    }

    // iterate over available devices and use the first that fits the requirements
    for (adapter_pair& pair : available_adapters)
    {
        std::string name = mi::examples::strings::wstr_to_str(pair.desc.Description);

        // create the device context
        if (SUCCEEDED(D3D12CreateDevice(
            pair.adapter.Get(), feature_level, _uuidof(D3DDevice), &m_device)))
        {
            // check ray tracing support
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 data;
            m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &data, sizeof(data));
            if (data.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
            {
                log_info("D3D Device does not support RTX: " + name);
                m_device = nullptr;
                continue;
            }

            // found a device that supports RTX
            log_info("Created context for D3D Device: " + name);
            found_adapter = true;
            break;
        }
        else
        {
            log_info("Failed to create D3D Device: " + name);
        }
    }

    if (!found_adapter)
    {
        log_error("No D3D device found that fits the requirements.");
        return false;
    }

    #if defined(_DEBUG)
    {
        ComPtr<ID3D12InfoQueue> pInfoQueue;
        if (SUCCEEDED(m_device.As(&pInfoQueue)))
        {
            pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
            pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);

            D3D12_MESSAGE_SEVERITY Severities[] =
            {
                D3D12_MESSAGE_SEVERITY_INFO
            };

            D3D12_INFO_QUEUE_FILTER NewFilter = {};
            NewFilter.DenyList.NumSeverities = _countof(Severities);
            NewFilter.DenyList.pSeverityList = Severities;

            if(log_on_failure(pInfoQueue->PushStorageFilter(&NewFilter),
                "Failed to setup D3D debug messages", SRC))
                return false;
        }
    }
    #endif

    // check if the device context is still valid
    if(log_on_failure(m_device->GetDeviceRemovedReason(),
       "Created device is in invalid state.", SRC))
       return false;

    // create a heap for all resources
    m_resource_descriptor_heap = new Descriptor_heap(
        this, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4096 /* hard coded */, "ResourceHeap");

    m_render_target_descriptor_heap = new Descriptor_heap(
        this, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 8 /* hard coded */, "RenderTargetHeap");

    // load the MDL SDK and check for success
    m_mdl_sdk = new Mdl_sdk(this);
    if (!m_mdl_sdk->is_valid())
    {
        log_error("MDL SDK not initialized properly.");
        return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------

void Base_application::update()
{
    m_update_args.scene_is_updating = m_scene_is_updating_next;
    m_render_args.scene_is_updating = m_scene_is_updating_next;

    // allow the application to adapt to new resolutions
    if ((m_window->get_width() != m_render_args.backbuffer_width) ||
        (m_window->get_height() != m_render_args.backbuffer_height))
    {
        m_render_args.backbuffer_width = m_window->get_width();
        m_render_args.backbuffer_height = m_window->get_height();
        on_resize(m_window->get_width(), m_window->get_height());
        flush_command_queues();
    }

    // compute elapsed time
    if (m_update_args.frame_number == 0)
    {
        m_mainloop_start_time = std::chrono::high_resolution_clock::now();
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        double new_total_time = (now - m_mainloop_start_time).count() * 1e-9;

        m_update_args.elapsed_time = new_total_time - m_update_args.total_time;
        m_update_args.total_time = new_total_time;
    }

    // update the application
    update(m_update_args);
}

// ------------------------------------------------------------------------------------------------

void Base_application::render()
{
    m_render_args.back_buffer = m_window->get_back_buffer();
    m_render_args.back_buffer_rtv = m_window->get_back_buffer_rtv();
    render(m_render_args);
    m_window->present_back_buffer();

    m_update_args.frame_number++;
    m_render_args.frame_number++;
}

}}} // mi::examples::mdl_d3d12
