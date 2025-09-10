/******************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Command_queue;
    class Descriptor_heap;
    class Texture;
    class IWindow;
    class Mdl_sdk;

    // --------------------------------------------------------------------------------------------

    /// Arguments passed to the render loop.
    struct Update_args
    {
        /// The number of the frame (update-render cycle) since application start.
        size_t frame_number;

        /// Time since last update in seconds.
        double elapsed_time;

        /// Total time in seconds since application start.
        double total_time;

        /// Indicates the application is currently updating the scene
        /// (geometry, materials, ...). This is set by the application
        /// and the application can skip updates and rendering in case.
        bool scene_is_updating;
    };

    // --------------------------------------------------------------------------------------------

    /// Arguments passed to the render loop.
    struct Render_args
    {
        /// Buffer that stores the rendering canvas.
        Texture* back_buffer;
        D3D12_CPU_DESCRIPTOR_HANDLE back_buffer_rtv;

        /// The number of the frame (update-render cycle) since application start.
        size_t frame_number;

        /// Width of the rendering canvas in pixels.
        size_t backbuffer_width;

        /// Height of the rendering canvas in pixels.
        size_t backbuffer_height;

        /// Indicates the application is currently updating the scene
        /// (geometry, materials, ...). This is set by the application
        /// and the application can skip updates and rendering in case.
        bool scene_is_updating;
    };

    // --------------------------------------------------------------------------------------------

    // Options to enable or disable certain features for instance because of GPU capabilities.
    struct Feature_options
    {
        /// Bindless resources for global access to all shaders and buffers.
        /// Available with Shader Model 6.6 and Resource Binding Tier 3
        /// see https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html
        bool HLSL_dynamic_resources = false;
    };

    // --------------------------------------------------------------------------------------------

    // Options that are defined at application start. They cannot be changed while the application
    // is running.
    class Base_options
    {
    public:
        // Supported slot modes for HLSL.
        enum Slot_mode {
            SM_NONE,
            SM_FIXED_1,
            SM_FIXED_2,
            SM_FIXED_4,
            SM_FIXED_8
        };

        /// Flags controlling the calculation of DF results.
        enum Df_flags {
            DF_FLAGS_NONE = 0,               ///< allows nothing -> black

            DF_FLAGS_ALLOW_REFLECT = 1,
            DF_FLAGS_ALLOW_TRANSMIT = 2,
            DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT = DF_FLAGS_ALLOW_REFLECT | DF_FLAGS_ALLOW_TRANSMIT,
            DF_FLAGS_ALLOWED_SCATTER_MODE_MASK = DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT,
        };

        struct Material_override {
            std::string selector;
            std::string material;
        };

        explicit Base_options()
            : features()
            , window_title(L"MDL D3D12 Example Application")
            , window_width(1280)
            , window_height(720)
            , mdl_paths()
            , mdl_next(false)
            , use_class_compilation(true)
            , fold_all_bool_parameters(false)
            , force_single_threading(false)
            , no_gui(false)
            , no_console_window(false)
            , hide_gui(false)
            , ray_depth(6)
            , sss_depth(256)
            , output_file("output.png")
            , lpe({ "beauty" })
            , iterations(1)
            , enable_auxiliary(true)
            , texture_results_cache_size(16)
            , automatic_derivatives(false)
            , handle_z_axis_up(false)
            , meters_per_scene_unit(1.0f)
            , uv_flip(true)
            , uv_scale(1.0f, 1.0f)
            , uv_offset(0.0f, 0.0f)
            , uv_repeat(false)
            , uv_saturate(false)
            , camera_pose_override(false)
            , camera_position(0.0f, 0.0f, 0.0f)
            , camera_focus(0.0f, 0.0f, 0.0f)
            , camera_fov(-1.0f)
            , gpu(-1)
            , gpu_debug(false)
            , enable_shader_cache(false)
            , shader_opt("O3")
            , distill_target("none")
            , distill_debug(false)
            , slot_mode(SM_NONE)
            , material_type("::material")
            , material_type_module("::%3Cbuiltins%3E")
            , aov_to_render("")
            , enable_bsdf_flags(false)
            , allowed_scatter_mode(DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT)
#if MDL_ENABLE_SLANG
            , use_slang(false)
#endif
#if MDL_ENABLE_MATERIALX
            , mtlx_to_mdl("latest")
            , materialxtest_mode(false)
#endif
        {
        }
        virtual ~Base_options() = default;

        Feature_options features;

        std::wstring window_title;
        size_t window_width;
        size_t window_height;
        std::vector<std::string> mdl_paths;
        bool mdl_next;
        bool use_class_compilation;
        bool fold_all_bool_parameters;
        bool force_single_threading;
        bool no_gui;
        bool no_console_window;
        bool hide_gui;
        size_t ray_depth;
        size_t sss_depth;
        std::string output_file;
        std::string generated_mdl_path;
        std::vector<std::string> lpe;
        size_t iterations;
        bool enable_auxiliary;
        size_t texture_results_cache_size;
        bool automatic_derivatives;
        bool handle_z_axis_up;
        float meters_per_scene_unit;
        bool uv_flip;
        DirectX::XMFLOAT2 uv_scale;
        DirectX::XMFLOAT2 uv_offset;
        bool uv_repeat;
        bool uv_saturate;
        bool camera_pose_override;
        DirectX::XMFLOAT3 camera_position;
        DirectX::XMFLOAT3 camera_focus;
        float camera_fov;
        int32_t gpu;
        bool gpu_debug;
        bool enable_shader_cache;
        std::string shader_opt;
        std::string distill_target;
        bool distill_debug;
        Slot_mode slot_mode;

        // with MDL 1.9, custom material types are supported
        std::string material_type; // the qualified name of the structure that defines the material
        std::string material_type_module; // the qualified name of the module that contains the type
        std::string aov_to_render; // field name of the custom material type to render

        bool enable_bsdf_flags;
        Df_flags allowed_scatter_mode;

#if MDL_ENABLE_SLANG
        bool use_slang;
#endif

#if MDL_ENABLE_MATERIALX
        std::vector<std::string> mtlx_paths;
        std::vector<std::string> mtlx_libraries;
        std::string mtlx_to_mdl;
        bool materialxtest_mode;
#endif
    };

    // --------------------------------------------------------------------------------------------

    class Base_dynamic_options
    {
    public:
        Base_dynamic_options(const Base_options* options);

        // Called by the application to check if the progressive rendering has to be restarted.
        // Returns true and resets in case the rendering has to be restarting. A further call to
        // this function will return false until a corresponding option changed.
        bool get_restart_progressive_rendering();

        /// Get/Set the currently active expression used by the renderer.
        /// Changes will trigger a restart of the progressive rendering.
        const std::string& get_active_lpe() const { return m_active_lpe; }
        void set_active_lpe(const std::string& expression)
        {
            if (m_active_lpe != expression)
            {
                m_active_lpe = expression;
                m_restart_progressive_rendering = true;
            }
        }

        /// Get/Set the available AOVs.
        /// Depends on the material type the renderer is set up to and the --aov option on start.
        const std::vector<std::string>& get_available_aovs() const { return m_available_aovs; }
        void set_available_aovs(const std::vector<std::string>& to_set)
        {
            m_available_aovs = to_set;
            m_active_aov = 0;
        }

        /// Get/Set the active AOV, i.e. the index into the available aov list.
        size_t get_active_aov() const { return m_active_aov; }
        void set_active_aov(size_t index)
        {
            if (m_active_aov != index)
            {
                m_active_aov = index; // no out of bounds check here
                m_restart_progressive_rendering = true;
            }
        }

    private:
        bool m_restart_progressive_rendering = true;
        std::string m_active_lpe;
        std::vector<std::string> m_available_aovs = {};
        size_t m_active_aov;
    };

    // --------------------------------------------------------------------------------------------

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

    // --------------------------------------------------------------------------------------------

    class Base_application
    {
        friend class Base_application_message_interface;

    public:
        virtual ~Base_application();

        /// called only from the main entry point of the application.
        int run(
            Base_options* options,
            Base_dynamic_options* dynamic_options,
            HINSTANCE hInstance,
            int nCmdShow);

        /// get access to the applications options that have been parsed from the command line
        const Base_options* get_options() const { return m_options; }

        /// get access to the application options that can change at runtime.
        Base_dynamic_options* get_dynamic_options() { return m_dynamic_options; }

        /// get access to the DXGI factory (required by the window)
        IDXGIFactory4* get_dxgi_factory() { return m_factory.Get(); }

        /// get access to the D3D device
        D3DDevice* get_device() { return m_device.Get(); }

        /// access to applications main window
        IWindow* get_window() const { return m_window; }

        /// get access to the applications command queues
        Command_queue* get_command_queue(D3D12_COMMAND_LIST_TYPE type);

        /// flush all command queues and make sure that all scheduled GPU work is done
        void flush_command_queues();

        /// heap for all resource views the application uses
        Descriptor_heap* get_resource_descriptor_heap();

        /// heap for all render targets the application uses
        Descriptor_heap* get_render_target_descriptor_heap();

        /// Access the MDL SDK.
        Mdl_sdk& get_mdl_sdk() { return *m_mdl_sdk; }

        /// Get the file path of the currently loaded scene.
        const std::string& get_scene_path() const { return m_scene_path; }

        /// Get access to a application wide time measuring tool.
        Profiling& get_profiling() { return m_profiling; }

    protected:
        explicit Base_application();

        virtual bool initialize(Base_options* options) { return true; }
        virtual bool load() = 0;
        virtual void update(const Update_args& args) = 0;
        virtual void render(const Render_args& args) = 0;
        virtual bool unload() = 0;
        virtual void key_down(uint8_t key) {};
        virtual void key_up(uint8_t key) {};

        /// allows the application to respond on window size changes.
        virtual void on_resize(size_t width, size_t height) = 0;

        /// Sets the 'scene_is_updating' property of the update and render arguments of the next
        /// Frame. The current frame is not affected.
        void set_scene_is_updating(bool value) { m_scene_is_updating_next = value; }

        /// Set the file path of the currently loaded scene.
        void set_scene_path(const std::string& file_path) { m_scene_path = file_path; }
    private:
        // allow the application to see / change options before loading
        bool initialize_internal(Base_options* options, Base_dynamic_options* dynamic_options);

        void update();
        void render();

        const Base_options* m_options;
        Base_dynamic_options* m_dynamic_options;
        std::string m_scene_path;

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
        bool m_scene_is_updating_next;

        Profiling m_profiling;
    };

}}} // mi::examples::mdl_d3d12
#endif
