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

// examples/mdl_sdk/dxr/mdl_d3d12/base_application.h

#ifndef MDL_D3D12_BASE_APPLICATION_H
#define MDL_D3D12_BASE_APPLICATION_H

#include "common.h"

namespace mdl_d3d12
{
    class Base_application;
    class Command_queue;
    class Descriptor_heap;
    class Texture;
    class IWindow;
    class Mdl_sdk;

    // arguments passed to the render loop
    struct Update_args
    {
        size_t frame_number;
        double elapsed_time;
        double total_time;
    };

    // arguments passed to the render loop
    struct Render_args
    {
        Texture* back_buffer;
        D3D12_CPU_DESCRIPTOR_HANDLE back_buffer_rtv;

        size_t frame_number;
        size_t backbuffer_width;
        size_t backbuffer_height;
    };

    class Base_options
    {
    public:
        explicit Base_options()
            : window_title(L"MDL D3D12 Example Application")
            , window_width(1280)
            , window_height(720)
            , mdl_paths()
            , use_class_compilation(true)
            , force_single_theading(false)
            , no_gui(false)
            , hide_gui(true)
            , gui_scale(1.0f)
            , ray_depth(6)
            , scene_directory(".")
            , output_file("output.png")
            , lpe("beauty")
            , iterations(1)
            , enable_auxiliary(true)
            , texture_results_cache_size(16)
            , automatic_derivatives(false)
            , handle_z_axis_up(false)
            , units_per_meter(1.0f)
            , gpu(-1)
        {
        }
        virtual ~Base_options() = default;
        std::wstring window_title;
        size_t window_width;
        size_t window_height;
        std::vector<std::string> mdl_paths;
        bool use_class_compilation;
        bool force_single_theading;
        bool no_gui;
        bool hide_gui;
        float gui_scale;
        size_t ray_depth;
        std::string scene_directory;
        std::string output_file;
        std::string lpe;
        size_t iterations;
        bool enable_auxiliary;
        size_t texture_results_cache_size;
        bool automatic_derivatives;
        bool handle_z_axis_up;
        float units_per_meter;
        int32_t gpu;

        std::unordered_map<std::string, std::string> user_options;
    };

    // connection between the windows or OS level message pump and the application
    class Base_application_message_interface final
    {
        friend class Base_application;
        explicit Base_application_message_interface(
            Base_application* app, 
            HINSTANCE instance);

    public:
        virtual ~Base_application_message_interface() = default;

        void key_down(uint8_t key);
        void key_up(uint8_t key);
        void paint();
        void resize(size_t width, size_t height, double dpi);

        Base_application* get_application() const { return m_app; }
        HINSTANCE get_win32_instance_handle() const { return m_instance; }

    private:
        Base_application* m_app;
        HINSTANCE m_instance;
    };


    class Base_application
    {
        friend class Base_application_message_interface;

    public:
        virtual ~Base_application();
        int run(Base_options* options, HINSTANCE hInstance, int nCmdShow);

        // get access the applications options that have been parsed from the command line
        const Base_options* get_options() const { return m_options; }

        // get access to the DXGI factory (required by the window)
        IDXGIFactory4* get_dxgi_factory() { return m_factory.Get(); }

        // get access to the D3D device
        D3DDevice* get_device() { return m_device.Get(); }

        // access to applications main window
        IWindow* get_window() const { return m_window; }

        // get access to the applications command queues
        Command_queue* get_command_queue(D3D12_COMMAND_LIST_TYPE type);

        // flush all command queues and make sure that all scheduled GPU work is done
        void flush_command_queues();

        // heap for all resource views the application uses
        Descriptor_heap* get_resource_descriptor_heap();

        // heap for all render targets the application uses
        Descriptor_heap* get_render_target_descriptor_heap();

        // access the MDL SDK
        Mdl_sdk& get_mdl_sdk() { return *m_mdl_sdk; }


    protected:
        explicit Base_application();

        virtual bool initialize(Base_options* options) { return true; }
        virtual bool load() = 0;
        virtual void update(const Update_args& args) = 0;
        virtual void render(const Render_args& args) = 0;
        virtual bool unload() = 0;
        virtual void key_down(uint8_t key) {};
        virtual void key_up(uint8_t key) {};

        // allows the application to respond on window size changes
        virtual void on_resize(size_t width, size_t height) = 0;

    private:
        // allow the application to see / change options before loading
        bool initialize_internal(Base_options* options);

        void update();
        void render();

        const Base_options* m_options;
        ComPtr<IDXGIFactory4> m_factory;
        ComPtr<D3DDevice> m_device;
        IWindow* m_window;

        Descriptor_heap* m_resource_descriptor_heap;
        Descriptor_heap* m_render_target_descriptor_heap;
        Mdl_sdk* m_mdl_sdk;

        std::unordered_map<D3D12_COMMAND_LIST_TYPE, Command_queue*> m_command_queues;

        Update_args m_update_args;
        Render_args m_render_args;
        std::chrono::steady_clock::time_point m_mainloop_start_time;
    };


} // mdl_d3d12

#endif
