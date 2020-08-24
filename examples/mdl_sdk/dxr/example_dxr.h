/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/dxr/example_dxr.h

#ifndef MDL_D3D12_EXAMPLE_DXR_H
#define MDL_D3D12_EXAMPLE_DXR_H

#include "mdl_d3d12/base_application.h"

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Camera_controls;
    struct Descriptor_heap_handle;
    template<typename T> class Dynamic_constant_buffer;
    class Environment;
    class Mdl_material;
    class Mdl_material_description;
    class Raytracing_pipeline;
    class Scene;
    class Scene_node;
    class Shader_binding_tables;
    class Texture;
}}}

namespace mi { namespace examples { namespace dxr
{
    class Example_dxr_options;
    class Info_overlay;
    class Gui_performance_overlay;
    class Gui_section_edit_material;
    enum class Example_dxr_gui_event : uint64_t;

    // --------------------------------------------------------------------------------------------

    // Different GUI windows that can be toggled using SPACE
    enum class Example_dxr_gui_mode : uint32_t
    {
        None =                  0,
        Main_gui =              (1 << 0),
        Performance_overlay =   (1 << 1),

        All = Main_gui | Performance_overlay
    };

    // --------------------------------------------------------------------------------------------

    // constants that are updated once per frame or stay constant
    // Note, make sure constant buffer elements are 4x32 bit aligned (important for vectors)
    struct Scene_constants
    {
        float total_time;
        float delta_time;

        /// (progressive) rendering
        uint32_t progressive_iteration;
        uint32_t max_ray_depth;
        uint32_t iterations_per_frame;

        /// tone mapping
        float exposure_compensation;
        float firefly_clamp_threshold;
        float burn_out;

        /// one additional point light for illustration
        uint32_t point_light_enabled;
        DirectX::XMFLOAT3 point_light_position;
        DirectX::XMFLOAT3 point_light_intensity;

        /// gamma correction while rendering to the frame buffer
        float output_gamma_correction;

        /// environment light
        float environment_intensity_factor;
        float environment_inv_integral;

        /// when auxiliary buffers are enabled, this index is used to select to one to display
        uint32_t display_buffer_index;

        /// resets the progressive iteration counter to restart the rendering in the next frame.
        void restart_progressive_rendering();

        /// Set the IBL constants when a new environment map is loaded.
        /// This also triggers an update of the firefly heuristic (if enabled).
        void update_environment(const mdl_d3d12::Environment* environment);

        /// set the firefly clamp threshold when light intensities changes.
        void update_firefly_heuristic(const mdl_d3d12::Environment* environment);
    };

    // --------------------------------------------------------------------------------------------

    enum class Ray_type
    {
        Radiance = 0,
        Shadow = 1,

        count,
    };

    // --------------------------------------------------------------------------------------------

    enum class Display_buffer_options
    {
        Beauty = 0,
        Albedo,
        Normal,

        count
    };

    // --------------------------------------------------------------------------------------------

    class Example_dxr : public mdl_d3d12::Base_application
    {
    public:
        explicit Example_dxr();

    public:
        /// Get the current scene, which contains the scene graph.
        mdl_d3d12::Scene* get_scene() { return m_scene; }

        /// Get the current environment used for IBL and as background.
        mdl_d3d12::Environment* get_environment() { return m_environment; }

    protected:
        bool initialize(mdl_d3d12::Base_options* options) final;
        bool load() final;

        // called once per frame before render is started
        void update(const mdl_d3d12::Update_args& args) final;

        // called once per frame after update is completed
        void render(const mdl_d3d12::Render_args& args) final;

        bool unload() final;

        // called on window resize
        void on_resize(size_t width, size_t height) final;

        void key_up(uint8_t key) final;

    private:

        /// Disposes the current scene and loads a new selected one.
        ///
        /// \param scene_path           absolute path the scene to load
        /// \param skip_pipeline_update do not update the pipeline if more loading
        ///                             steps are required or needed before the
        ///                             pipeline can be created
        bool load_scene(
            const std::string& scene_path,
            bool skip_pipeline_update = false);

        /// Disposes the current environment and loads a new selected one.
        ///
        /// \param environment_path     absolute path the environment map to load
        /// \param skip_pipeline_update do not update the pipeline if more loading
        ///                             steps are required or needed before the
        ///                             pipeline can be created
        bool load_environment(
            const std::string& environment_path,
            bool skip_pipeline_update = false);

        /// Replace the selected material.
        /// Setting a new description will change the material definition along with
        /// the instance, the compiled material, and the generated code.
        void replace_material(
            mdl_d3d12::Mdl_material* material,
            const mdl_d3d12::Mdl_material_description& description,
            Gui_section_edit_material* mat_gui);

        /// Reload the current material. The definition will be updated
        /// in case it changed on disk.
        void reload_material(
            mdl_d3d12::Mdl_material* material,
            Gui_section_edit_material* mat_gui);

        /// Recompile the current material without reloading the module from disk.
        /// This is required after parameter changes in instance compilation mode or
        /// when the structure of the expression graph attached to a material changed.
        void recompile_materials(
            mdl_d3d12::Mdl_material* selected_material,
            bool recompile_only_the_selected_material,
            Gui_section_edit_material* mat_gui);

        bool update_material_pipeline();
        bool update_rendering_pipeline();

        mdl_d3d12::Texture* m_frame_buffer;
        mdl_d3d12::Descriptor_heap_handle m_frame_buffer_uav;

        mdl_d3d12::Texture* m_output_buffer;
        mdl_d3d12::Descriptor_heap_handle m_output_buffer_uav;

        mdl_d3d12::Texture* m_albedo_buffer;
        mdl_d3d12::Descriptor_heap_handle m_albedo_buffer_uav;

        mdl_d3d12::Texture* m_normal_buffer;
        mdl_d3d12::Descriptor_heap_handle m_normal_buffer_uav;

        mdl_d3d12::Descriptor_heap_handle m_acceleration_structure_srv;
        mdl_d3d12::Raytracing_pipeline* m_pipeline[2];
        mdl_d3d12::Shader_binding_tables* m_shader_binding_table[2];
        size_t m_active_pipeline_index; // pipeline and binding tables can be swapped after updates
        bool m_swap_next_frame;

        mdl_d3d12::Scene* m_scene;
        mdl_d3d12::Dynamic_constant_buffer<Scene_constants>* m_scene_constants;
        mdl_d3d12::Descriptor_heap_handle m_scene_cbv;

        mdl_d3d12::Camera_controls* m_camera_controls;
        mdl_d3d12::Descriptor_heap_handle m_camera_cbv;

        mdl_d3d12::Environment* m_environment;

        bool m_take_screenshot;
        bool m_toggle_fullscreen;

        Example_dxr_gui_mode m_gui_mode;
        Gui_performance_overlay* m_main_window_performance_overlay;
        Info_overlay* m_info_overlay;
    };

}}} // mi::examples::mdl_d3d12
#endif
