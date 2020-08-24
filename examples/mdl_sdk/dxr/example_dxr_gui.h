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

// examples/mdl_sdk/dxr/example_dxr_gui.h

#ifndef MDL_D3D12_EXAMPLE_DXR_GUI_H
#define MDL_D3D12_EXAMPLE_DXR_GUI_H

#include "mdl_d3d12/common.h"
#include <gui/gui.h>
#include <gui/gui_material_properties.h>

namespace mi { namespace neuraylib
{
    class ITransaction;
}}

namespace mi { namespace examples { namespace mdl_d3d12
{
    class Base_application;
    class Camera_controls;
    class Mdl_material;
    class Scene;
    struct Update_args;
}}}

namespace mi { namespace examples { namespace dxr
{
    struct Scene_constants;
    class Example_dxr;
    class Example_dxr_options;

    // --------------------------------------------------------------------------------------------

    enum class Example_dxr_gui_event : uint64_t
    {
        Replace_current_material = 1,
        Reload_current_material,
        Recompile_current_material,
        Recompile_all_materials,

        Menu_file_open_scene,
        Menu_file_open_environment,
        Menu_file_save_screenshot,
        Menu_file_quit,
    };

    // --------------------------------------------------------------------------------------------

    class Gui_section_rendering : public mi::examples::gui::Section
    {
    public:
        explicit Gui_section_rendering(
            Example_dxr* app,
            mi::examples::gui::Root* gui,
            Scene_constants* scene_data,
            const Example_dxr_options* options);

    protected:
        void update(mi::neuraylib::ITransaction* transaction) final;

    private:
        Example_dxr* m_app;
        Scene_constants* m_scene_data;
        const Example_dxr_options* m_options;
        uint32_t m_default_output_buffer_index;
        bool m_enable_firefly_clamping;
    };

    // --------------------------------------------------------------------------------------------

    class Gui_section_camera : public mi::examples::gui::Section
    {
    public:
        explicit Gui_section_camera(
            Example_dxr* app,
            mi::examples::gui::Root* gui,
            Scene_constants* scene_data,
            mdl_d3d12::Camera_controls* camera_controls);

    protected:
        void update(mi::neuraylib::ITransaction* transaction) final;

    private:
        Example_dxr* m_app;
        Scene_constants* m_scene_data;
        mdl_d3d12::Camera_controls* m_camera_controls;
    };

    // --------------------------------------------------------------------------------------------

    class Gui_section_light : public mi::examples::gui::Section
    {
    public:
        explicit Gui_section_light(
            Example_dxr* app,
            mi::examples::gui::Root* gui,
            Scene_constants* scene_data,
            const Example_dxr_options* options);

    protected:
        void update(mi::neuraylib::ITransaction* transaction) final;

    private:
        Example_dxr* m_app;
        Scene_constants* m_scene_data;
        const Example_dxr_options* m_options;
        bool m_group_environment;
        bool m_group_point_light;
        bool m_enable_point_light;

        DirectX::XMFLOAT3 m_point_light_color;
        float m_point_light_intensity;

        DirectX::XMFLOAT3 m_default_point_light_color;
        float m_default_point_light_intensity;
    };

    // --------------------------------------------------------------------------------------------

    class Gui_section_mdl_options : public mi::examples::gui::Section
    {
    public:
        explicit Gui_section_mdl_options(
            Example_dxr* app,
            mi::examples::gui::Root* gui,
            const Example_dxr_options* options);

    protected:
        void update(mi::neuraylib::ITransaction* transaction) final;

    private:
        Example_dxr* m_app;
        const Example_dxr_options* m_options;
        bool m_group_class_compilation;
        bool m_created_shader_cache_folder;
    };

    // --------------------------------------------------------------------------------------------

    /// This section extends the material section provided in the example shared project.
    /// The extension (by composition) allows to select materials, reload and replace Materials,
    /// and implements the resource handler. All those aspects are application dependent and
    /// therefore implemented here.
    class Gui_section_edit_material
        : public mi::examples::gui::Section
        , private mi::examples::gui::Section_material_resource_handler
    {
    public:
        explicit Gui_section_edit_material(
            Example_dxr* app,
            mi::examples::gui::Root* gui,
            Scene_constants* scene_data);

        ~Gui_section_edit_material() = default;

        void update_material_list();
        void unbind_material();
        bool bind_material(mdl_d3d12::Mdl_material* mat);
        mdl_d3d12::Mdl_material* get_bound_material();

    protected:
        void update(mi::neuraylib::ITransaction* transaction) final;

    private:
        mi::examples::gui::Section_material m_internal_section;

        Example_dxr* m_app;
        Scene_constants* m_scene_data;
        std::vector<mdl_d3d12::Mdl_material*> m_scene_materials;
        std::vector<std::string> m_scene_materials_names;
        uint32_t m_bound_material_index;

    private: // mi::examples::gui::Section_material_resource_handler

        void init_resource_handling(mdl_d3d12::Mdl_material* material);

        mi::Size get_available_resource_count(
            mi::neuraylib::IValue::Kind kind) override;

        mi::Uint32 get_available_resource_id(
            mi::neuraylib::IValue::Kind kind,
            mi::Size index) override;

        const char* get_available_resource_name(
            mi::neuraylib::IValue::Kind kind,
            mi::Size index) override;

        mi::Uint32 get_available_resource_id(
            mi::neuraylib::IValue::Kind kind,
            const char* db_name) override;

        std::vector<std::pair<mi::Uint32, std::string>> m_texture_2ds;
        // std::vector<std::pair<mi::Uint32, std::string>> m_light_profiles;
        // std::vector<std::pair<mi::Uint32, std::string>> m_mbsdfs;

        std::vector<char> m_last_assign_input_buffer;
        std::string m_last_assign_new_material_name;
        bool m_selected_material_supports_reloading;
        bool m_wait_for_external_popup;
        bool m_stop_waiting_for_external_popup;
        bool m_keep_open_after_external_popup_closed;
    };

    // --------------------------------------------------------------------------------------------

    class Gui_performance_overlay : public mi::examples::gui::Base_element
    {
    public:
        explicit Gui_performance_overlay(mi::examples::gui::Root* gui);
        ~Gui_performance_overlay() = default;

        /// special update function that is called directly by the application.
        void update(
            const mdl_d3d12::Update_args& args,
            const Scene_constants& scene_data);

    private:
        /// Unused. Would be called when added to a panel.
        void update(mi::neuraylib::ITransaction* /*transaction*/) final { }
    };

    // --------------------------------------------------------------------------------------------

    class Info_overlay : public mi::examples::gui::Base_element
    {
    public:
        explicit Info_overlay(mi::examples::gui::Root* gui);
        ~Info_overlay() = default;

        /// special update function that is called directly by the application.
        void update(const char* text);

    private:
        /// Unused. Would be called when added to a panel.
        void update(mi::neuraylib::ITransaction* /*transaction*/) final { }
    };
}}}

#endif
