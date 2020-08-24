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

#include "example_dxr_gui.h"

#include "example_dxr.h"
#include "example_dxr_options.h"
#include "mdl_d3d12/camera_controls.h"
#include "mdl_d3d12/mdl_d3d12.h"
#include "mdl_d3d12/scene.h"
#include <gui/gui.h>
#include <gui/gui_material_properties.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace mi { namespace examples { namespace dxr
{
using namespace mi::examples::mdl_d3d12;


Gui_section_rendering::Gui_section_rendering(
    Example_dxr* app,
    mi::examples::gui::Root* gui,
    Scene_constants* scene_data,
    const Example_dxr_options* options)
    : mi::examples::gui::Section(gui, "Rendering", true)
    , m_app(app)
    , m_scene_data(scene_data)
    , m_options(options)
    , m_enable_firefly_clamping(scene_data->firefly_clamp_threshold > 0.0f)
{
    if (m_options->lpe == "albedo")
        m_default_output_buffer_index = static_cast<uint32_t>(Display_buffer_options::Albedo);
    else if (m_options->lpe == "normal")
        m_default_output_buffer_index = static_cast<uint32_t>(Display_buffer_options::Normal);
    else
        m_default_output_buffer_index = static_cast<uint32_t>(Display_buffer_options::Beauty);
}

// ------------------------------------------------------------------------------------------------

void Gui_section_rendering::update(mi::neuraylib::ITransaction* /*transaction*/)
{
    using Gui_control = mi::examples::gui::Control;

    float default_0 = 0.0f;
    bool default_false = false;

    uint32_t default_value = uint32_t(m_options->ray_depth);
    if (Gui_control::slider("Path length", "Maximum ray path length.",
        &m_scene_data->max_ray_depth, &default_value, Gui_control::Flags::None, 1u, 16u))
        m_scene_data->restart_progressive_rendering();

    bool vsync = m_app->get_window()->get_vsync();
    if (Gui_control::checkbox(
        "Vsync", "Limits the number of iterations to refresh rate of the display.",
        &vsync, &default_false, Gui_control::Flags::None))
        m_app->get_window()->set_vsync(vsync);

    Gui_control::slider(
        "Exposure", "Brightness adjustment of the output image. [in stops]",
        &m_scene_data->exposure_compensation, &default_0, Gui_control::Flags::None, -3.f, 3.f);

    Gui_control::slider(
        "Burnout", "Tone mapping parameter. Increases the brightness of highlights.",
        &m_scene_data->burn_out, &m_options->tone_mapping_burn_out,
        Gui_control::Flags::None, 0.0f, 1.0f);

    if (Gui_control::selection<uint32_t>(
        "Output Buffer", "Rendering result, albedo, or normal buffer.",
        &m_scene_data->display_buffer_index, &m_default_output_buffer_index,
        Gui_control::Flags::None, [](uint32_t i)
        {
            switch (i)
            {
            case 0: return "Beauty";
            case 1: return "Albedo";
            case 2: return "Normal";
            default: return (const char*) nullptr;
            }
        }))
    {
        m_scene_data->restart_progressive_rendering();
    }

    if (Gui_control::checkbox(
        "Firefly Filter", "Clamps high spikes in intensity. "
        "Introduces an error but improves the visual convergence rate.",
        &m_enable_firefly_clamping, &m_options->firefly_clamp, Gui_control::Flags::None))
    {
        if (m_enable_firefly_clamping)
        {
            m_scene_data->firefly_clamp_threshold = 0.0f;
            m_scene_data->update_firefly_heuristic(m_app->get_environment());
        }
        else
        {
            m_scene_data->firefly_clamp_threshold = -1.0f;
            m_scene_data->restart_progressive_rendering();
        }
    }
}

// ------------------------------------------------------------------------------------------------

Gui_section_camera::Gui_section_camera(
    Example_dxr* app,
    mi::examples::gui::Root* gui,
    Scene_constants* scene_data,
    mdl_d3d12::Camera_controls* camera_controls)
    : mi::examples::gui::Section(gui, "Camera", true)
    , m_app(app)
    , m_scene_data(scene_data)
    , m_camera_controls(camera_controls)
{
}

// ------------------------------------------------------------------------------------------------

void Gui_section_camera::update(mi::neuraylib::ITransaction* /*transaction*/)
{
    float default_1_0 = 1.0f;
    float default_fov = 45.0f;
    const float* no_float = nullptr;

    mi::examples::gui::Control::drag("Movement Speed", "use CTRL + Left-Click to set explicit values.",
        &m_camera_controls->movement_speed, &default_1_0,
        mi::examples::gui::Control::Flags::None, 0.0f, 50.0f);

    mi::examples::gui::Control::drag("Rotation Speed", "use CTRL + Left-Click to set explicit values.",
        &m_camera_controls->rotation_speed, &default_1_0,
        mi::examples::gui::Control::Flags::None, 0.0f, 5.0f);

    Camera* cam = m_camera_controls->get_target()->get_camera();
    float fov = cam->get_field_of_view() * mdl_d3d12::ONE_OVER_PI * 180.0f;
    float len = cam->get_focal_length();

    if (mi::examples::gui::Control::drag("Field of View", "Vertical field of view in degrees.",
        &fov, &default_fov, mi::examples::gui::Control::Flags::None, 1.0f, 179.0f, 0.1f, "%.2f deg"))
    {
        cam->set_field_of_view(fov * mdl_d3d12::PI / 180.0f);
        m_scene_data->restart_progressive_rendering();
    }

    if (mi::examples::gui::Control::drag(
        "Focal Length", "Assuming traditional 35mm film, measuring 36x24mm.",
        &len, no_float, mi::examples::gui::Control::Flags::None, 1.0f, 200.f, 0.1f, "%.2f mm"))
    {
        cam->set_focal_length(len);
        m_scene_data->restart_progressive_rendering();
    }

    if (mi::examples::gui::Control::button("", "Fit all scene objects",
        "Sets the camera to be able to see all objects in the scene.",
        mi::examples::gui::Control::Flags::None))
    {
        const Bounding_box& aabb = m_app->get_scene()->get_root()->get_global_bounding_box();
        m_camera_controls->fit_into_view(aabb);
        m_scene_data->restart_progressive_rendering();
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Gui_section_light::Gui_section_light(
    Example_dxr* app,
    mi::examples::gui::Root* gui,
    Scene_constants* scene_data,
    const Example_dxr_options* options)
    : mi::examples::gui::Section(gui, "Light/Environment", true)
    , m_app(app)
    , m_scene_data(scene_data)
    , m_options(options)
    , m_group_environment(true)
    , m_group_point_light(true)
    , m_enable_point_light(m_scene_data->point_light_enabled == 1u)
{
    // split intensity vector into color and scalar intensity
    m_point_light_color = scene_data->point_light_intensity;
    m_point_light_intensity = maximum(scene_data->point_light_intensity);
    if (m_point_light_intensity > 0.0f)
        m_point_light_color /= m_point_light_intensity;

    m_default_point_light_color = options->point_light_intensity;
    m_default_point_light_intensity = maximum(options->point_light_intensity);
    if (m_default_point_light_intensity > 0.0f)
        m_default_point_light_color /= m_default_point_light_intensity;
}

// ------------------------------------------------------------------------------------------------

void Gui_section_light::update(mi::neuraylib::ITransaction* /*transaction*/)
{
    mi::examples::gui::Control::group<bool>("Environment", m_group_environment, [&]
        {
            if (mi::examples::gui::Control::drag("Intensity",
                "Scale Factor applied the environment map.",
                &m_scene_data->environment_intensity_factor, &m_options->hdr_scale,
                mi::examples::gui::Control::Flags::None, 0.0f))
            {
                m_scene_data->update_environment(m_app->get_environment());
            }
            return false; // not relevant here
        });

    mi::examples::gui::Control::group<bool>("Point Light", m_group_point_light, [&]
        {
            if (mi::examples::gui::Control::checkbox(
                "Enabled", "",
                &m_enable_point_light, &m_options->point_light_enabled,
                mi::examples::gui::Control::Flags::None))
            {
                m_scene_data->point_light_enabled = m_enable_point_light ? 1u : 0u;
                m_scene_data->update_firefly_heuristic(m_app->get_environment());
            }

            if (mi::examples::gui::Control::drag("Intensity", "Intensity of the point light.",
                    &m_point_light_intensity, &m_default_point_light_intensity,
                    mi::examples::gui::Control::Flags::None,
                    0.0f, std::numeric_limits<float>::max(), 10.0f)
                ||
                mi::examples::gui::Control::pick("Color", "Emission color of the point light.",
                    &m_point_light_color.x, &m_default_point_light_color.x,
                    mi::examples::gui::Control::Flags::None))
            {
                m_scene_data->point_light_intensity = m_point_light_color * m_point_light_intensity;
                m_scene_data->update_firefly_heuristic(m_app->get_environment());
            }
            return false; // not relevant here
        });
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Gui_section_mdl_options::Gui_section_mdl_options(
    Example_dxr* app,
    mi::examples::gui::Root* gui,
    const Example_dxr_options* options)
    : mi::examples::gui::Section(gui, "MDL Options", true)
    , m_app(app)
    , m_options(options)
    , m_group_class_compilation(true)
    , m_created_shader_cache_folder(false)
{
}

// ------------------------------------------------------------------------------------------------

void Gui_section_mdl_options::update(mi::neuraylib::ITransaction* /*transaction*/)
{
    mi::examples::mdl_d3d12::Mdl_sdk::Options& mdl_options =
        m_app->get_mdl_sdk().get_options();

    mi::examples::gui::Control::group<bool>("Class Compilation", m_group_class_compilation, [&]
        {
            bool recompile_materials = false;
            const bool default_false = false;

            if (mi::examples::gui::Control::checkbox("Enabled",
                "Allow for interactive parameter editing. If disabled, Instance compilation is "
                "used which aims for best performance.",
                &mdl_options.use_class_compilation, &m_options->use_class_compilation,
                mi::examples::gui::Control::Flags::None))
                    recompile_materials = true;

            if (mi::examples::gui::Control::checkbox("Fold Booleans",
                "Performance optimization that bakes the value of boolean parameters into the "
                "material. This improves compile and runtime performance but editing a boolean "
                "will be slow as it requires recompilation.",
                &mdl_options.fold_all_bool_parameters, &default_false,
                mdl_options.use_class_compilation
                    ? mi::examples::gui::Control::Flags::None
                    : mi::examples::gui::Control::Flags::Disabled))
                        recompile_materials = true;

            if (mi::examples::gui::Control::checkbox("Fold Enums",
                "Performance optimization that bakes the value of enum parameters into the "
                "material. This improves compile and runtime performance but editing an enum "
                "will be slow as it requires recompilation.",
                &mdl_options.fold_all_enum_parameters, &default_false,
                mdl_options.use_class_compilation
                    ? mi::examples::gui::Control::Flags::None
                    : mi::examples::gui::Control::Flags::Disabled))
                        recompile_materials = true;

            // notify the application
            if (recompile_materials)
                create_event(static_cast<mi::Size>(Example_dxr_gui_event::Recompile_all_materials));

            return false; // not relevant here
        });

    if (mi::examples::gui::Control::checkbox("Enable Shader Cache",
        "Performance optimization on application side (with support of the SDK). Generated code is "
        "stored to disk when first compiled with this option enabled. Repeated usage of the same "
        "generated code restores the cached data if it is still valid.",
        &mdl_options.enable_shader_cache, &m_options->enable_shader_cache,
        mi::examples::gui::Control::Flags::None))
    {
        if (!m_created_shader_cache_folder)
        {
            m_created_shader_cache_folder = true;
            mi::examples::io::mkdir(mi::examples::io::get_executable_folder() + "/shader_cache");
        }
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Gui_section_edit_material::Gui_section_edit_material(
    Example_dxr* app,
    mi::examples::gui::Root* gui,
    Scene_constants* scene_data)
    : mi::examples::gui::Section(gui, "Material", false)
    , m_internal_section(
        gui,
        "Material",
        false,
        &app->get_mdl_sdk().get_evaluator(),
        &app->get_mdl_sdk().get_factory())
    , m_app(app)
    , m_scene_data(scene_data)
    , m_scene_materials()
    , m_bound_material_index(0)
    , m_last_assign_input_buffer(4096, '\0')
    , m_last_assign_new_material_name("::dxr::not_available")
    , m_selected_material_supports_reloading(true)
    , m_wait_for_external_popup(false)
    , m_stop_waiting_for_external_popup(false)
{
    std::string override_material;
    if (app->get_options()->get_user_options("override_material", override_material))
        m_last_assign_new_material_name = override_material;

    memcpy(
        m_last_assign_input_buffer.data(),
        m_last_assign_new_material_name.data(),
        m_last_assign_new_material_name.size());

    update_material_list();
}

// ------------------------------------------------------------------------------------------------

void Gui_section_edit_material::update_material_list()
{
    Mdl_material* current = m_bound_material_index < m_scene_materials.size()
        ? m_scene_materials[m_bound_material_index] : nullptr;

    // setup or update the material selection
    Mdl_material_library& mat_lib = *m_app->get_mdl_sdk().get_library();
    m_scene_materials.clear();
    m_scene_materials.push_back(nullptr);
    m_scene_materials_names.clear();
    m_scene_materials_names.push_back("[none]");

    mat_lib.visit_materials([&](Mdl_material* mat) {
        m_scene_materials.push_back(mat);
        m_scene_materials_names.push_back(mat->get_name());
        return true; // continue visits
        });

    // if there is only one material select it (at least until we have picking)
    if (current == nullptr && m_scene_materials.size() == 2)
        current = m_scene_materials[1];

    // try to select the already selected material again
    bind_material(current);
}

// ------------------------------------------------------------------------------------------------

void Gui_section_edit_material::unbind_material()
{
    m_bound_material_index = 0;
    init_resource_handling(nullptr);
    m_internal_section.unbind_material();
}

// ------------------------------------------------------------------------------------------------

bool Gui_section_edit_material::bind_material(Mdl_material* mat)
{
    if (!mat)
    {
        unbind_material();
        return true;
    }

    for (uint32_t i = 0, n = m_scene_materials.size(); i < n; ++i)
        if (m_scene_materials[i] == mat)
        {
            m_bound_material_index = i;
            m_app->get_mdl_sdk().get_transaction().execute<void>(
                [&](mi::neuraylib::ITransaction* t)
                {
                    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_mat(
                        t->access<const mi::neuraylib::ICompiled_material>(
                            mat->get_material_compiled_db_name().c_str()));

                    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                        mat->get_target_code()->get_generated_target());

                    mi::base::Handle<const mi::neuraylib::ITarget_value_layout> layout(
                        mat->get_argument_layout());

                    init_resource_handling(mat);

                    // When using class compilation, the argument block and its
                    // layout is required for mapping the available class parameters
                    // to GUI elements directly. this allows changing parameters without
                    // recompiling the material instances und updating the generated code.
                    m_internal_section.bind_material(
                        t, mat->get_material_instance_db_name().c_str(),
                        compiled_mat.get(), target_code.get(),
                        layout.get(), (char*)mat->get_argument_data(), this);

                    m_selected_material_supports_reloading =
                        mat->get_material_desciption().supports_reloading();
                });

            return true;
        }
    return false;
}

// ------------------------------------------------------------------------------------------------

mdl_d3d12::Mdl_material* Gui_section_edit_material::get_bound_material()
{
    return m_bound_material_index < m_scene_materials.size()
        ? m_scene_materials[m_bound_material_index] : nullptr;
}

// ------------------------------------------------------------------------------------------------

namespace
{
    static bool __mdl_browser_available_tested = false;
    static std::string __mdl_browser_available_path = "";
    bool is_mdl_browser_available()
    {
        if (__mdl_browser_available_tested)
            return !__mdl_browser_available_path.empty();
        __mdl_browser_available_tested = true;
        __mdl_browser_available_path = "";

        // first try, the build folder relative to this example
        std::string mdl_browser_path = mi::examples::strings::replace(
            mi::examples::io::get_executable_folder(),
            "/mdl_sdk/dxr/", "/mdl_sdk/mdl_browser/mdl_browser/") + "/mdl_browser.exe";
        if (mi::examples::io::file_exists(mdl_browser_path))
        {
            __mdl_browser_available_path = mdl_browser_path;
            return true;
        }

        // second try, the current executable directory
        mdl_browser_path = mi::examples::io::get_executable_folder() + "/mdl_browser.exe";
        if (mi::examples::io::file_exists(mdl_browser_path))
        {
            __mdl_browser_available_path = mdl_browser_path;
            return true;
        }
        return false;
    }

    const std::string& get_mdl_browser_path()
    {
        if (!__mdl_browser_available_tested)
            is_mdl_browser_available();
        return __mdl_browser_available_path;
    }

    std::string open_mdl_browser(mi::neuraylib::IMdl_configuration* config)
    {
        std::string browser_path = get_mdl_browser_path();
        if (browser_path.empty())
            return "";

        // create the command line with mdl paths
        std::string cmd = mi::examples::strings::replace(browser_path, " ", "\" \"");
        for (mi::Size i = 0, n = config->get_mdl_paths_length(); i < n; ++i)
        {
            mi::base::Handle<const mi::IString> sp(config->get_mdl_path(i));
            std::string so_s = mi::examples::io::normalize(sp->get_c_str());
            cmd += " --mdl_path \"" + so_s + "\"";
        }
        cmd += " --no_qt_mode"; // dxr is not a qt application

        // run the browser
        std::vector<char> buffer(128);
        std::string result;
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
        bool was_open = false;
        while (pipe && fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
            was_open = true;
        }
        if (!was_open) {
            log_error("Opening MDL Browser failed: " + browser_path);
            return "";
        }

        // evaluate the result
        if (!mi::examples::strings::starts_with(result, "::"))
        {
            std::string msg = "MDL Browser exited without a selected material: " + browser_path;
            if (result.empty())
                msg += "\n" + result;
            log_error(msg);
            return "";
        }

        // valid material
        return result;
    }
}

// ------------------------------------------------------------------------------------------------

void Gui_section_edit_material::update(mi::neuraylib::ITransaction* transaction)
{
    uint32_t default_section = 0;
    if (mi::examples::gui::Control::selection("Selected material\n", "Material selected for editing",
        &m_bound_material_index, &default_section, mi::examples::gui::Control::Flags::None,
        m_scene_materials_names))
            bind_material(m_scene_materials[m_bound_material_index]);

    ImGui::Spacing();

    if (m_bound_material_index != 0)
    {
        if (mi::examples::gui::Control::button(
            "", "Assign new material...",
            "Replaces the current material with a new one.",
            mi::examples::gui::Control::Flags::None))
        {
            ImGui::OpenPopup("Select Material");
        }

        ImGui::SetNextWindowSize(ImVec2(400, -1), ImGuiCond_Once);
        if (ImGui::BeginPopupModal("Select Material", NULL))
        {
            float width = ImGui::GetContentRegionAvail().x;

            ImGui::Spacing();
            ImGui::TextWrapped("Please specify the material to load using a qualified material "
                "name or the path of an MDLE file.");
            ImGui::Spacing();

            if (m_wait_for_external_popup)
            {
                if (m_stop_waiting_for_external_popup)
                {
                    if(!m_keep_open_after_external_popup_closed)
                        ImGui::CloseCurrentPopup();

                    m_wait_for_external_popup = false;
                    m_stop_waiting_for_external_popup = false;
                }

                const char* text = "Waiting for the dialog to close...";

                ImVec2 available = ImGui::GetContentRegionAvail();
                ImVec2 text_size = ImGui::CalcTextSize(text, nullptr, false, available.x);

                ImVec2 pos = ImGui::GetCursorPos();
                pos.x += (available.x - text_size.x) * 0.5f;
                pos.y += (available.y - text_size.y) * 0.5f;

                ImGui::SetCursorPos(pos);
                ImGui::TextWrapped(text);
            }
            else
            {
                bool file_picker_available = true;
                bool mdl_browser_available = is_mdl_browser_available();

                const char* browse_button_text = "browse...";
                float browse_button_width;
                if (mdl_browser_available)
                {
                    browse_button_width = ImGui::CalcTextSize(browse_button_text).x + 16.f;
                    width = width - browse_button_width - ImGui::GetStyle().ItemSpacing.x;
                }

                const char* file_picker_text = "pick...";
                float file_picker_width;
                if (file_picker_available)
                {
                    file_picker_width = ImGui::CalcTextSize(file_picker_text).x + 16.f;
                    width = width - file_picker_width - ImGui::GetStyle().ItemSpacing.x;
                }

                // executed when hitting the OK-button or when hitting Enter in the text field.
                auto commit_action = [&]()
                {
                    ImGui::CloseCurrentPopup();
                    m_last_assign_new_material_name = std::string(m_last_assign_input_buffer.data());
                    create_event(
                        static_cast<mi::Size>(Example_dxr_gui_event::Replace_current_material),
                        m_last_assign_new_material_name.data());
                };

                ImGui::SetNextItemWidth(width);
                if (ImGui::InputText("##hidden",
                    m_last_assign_input_buffer.data(), m_last_assign_input_buffer.size() - 1,
                    ImGuiInputTextFlags_EnterReturnsTrue))
                        commit_action();

                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextUnformatted("A qualified material name has the form: "
                        "[::<package>]::<module>::<material>\n"
                        "A name that is not matching this pattern will be "
                        "tested against other naming conventions like MDLE file names. If none of "
                        "them is matching, a GLTF support material will be used. "
                        "The latter can also be used to remove a material that was assigned "
                        "or loaded before");
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }

                if (mdl_browser_available)
                {
                    ImGui::SameLine();
                    if (ImGui::Button(browse_button_text, ImVec2(browse_button_width, 0.f)))
                    {
                        m_wait_for_external_popup = true;
                        std::thread([&]() {
                            // open the browser
                            std::string material_file_path = open_mdl_browser(
                                &m_app->get_mdl_sdk().get_config());

                            // material selection (or no error)
                            if (!material_file_path.empty())
                            {
                                // raise an event to notify the application
                                m_last_assign_new_material_name = material_file_path;
                                m_keep_open_after_external_popup_closed = false;

                                create_event(
                                    static_cast<mi::Size>(
                                        Example_dxr_gui_event::Replace_current_material),
                                    m_last_assign_new_material_name.data());
                            }
                            else
                                m_keep_open_after_external_popup_closed = true;
                            m_stop_waiting_for_external_popup = true;
                        }).detach();
                    }
                }
                if (file_picker_available)
                {
                    ImGui::SameLine();
                    if (ImGui::Button(file_picker_text, ImVec2(file_picker_width, 0.f)))
                    {
                        m_wait_for_external_popup = true;
                        std::thread([&]() {
                            // open the file picker
                            mi::examples::io::open_file_name_dialog dialog;
                            dialog.add_type("Encapsulated MDL", "mdle");

                            // check for code generators that match this name
                            m_app->get_mdl_sdk().get_library()->visit_material_description_loaders(
                            [&](const IMdl_material_description_loader* loader)
                            {
                                for (size_t i = 0, n = loader->get_file_type_count(); i < n; ++i)
                                    dialog.add_type(
                                        loader->get_file_type_description(i),
                                        loader->get_file_type_extension(i));
                                return true; // continue visits
                            });

                            std::string material_file_path = dialog.show();

                            // material selection (or no error)
                            if (!material_file_path.empty())
                            {
                                // raise an event to notify the application
                                m_last_assign_new_material_name = material_file_path;
                                m_keep_open_after_external_popup_closed = false;

                                create_event(
                                    static_cast<mi::Size>(
                                        Example_dxr_gui_event::Replace_current_material),
                                    m_last_assign_new_material_name.data());
                            }
                            else
                                m_keep_open_after_external_popup_closed = true;
                            m_stop_waiting_for_external_popup = true;
                        }).detach();
                    }
                }

                ImGui::Spacing();
                ImGui::Spacing();

                if (ImGui::Button("OK", ImVec2(120.f, 0.f)))
                    commit_action();
                ImGui::SetItemDefaultFocus();
                ImGui::SameLine();
                if (ImGui::Button("Cancel", ImVec2(120, 0)))
                {
                    ImGui::CloseCurrentPopup();
                    memset(m_last_assign_input_buffer.data(), '\0', m_last_assign_input_buffer.size());
                    memcpy(
                        m_last_assign_input_buffer.data(),
                        m_last_assign_new_material_name.data(),
                        m_last_assign_new_material_name.size());
                }
            }
            ImGui::EndPopup();
        }

        if (mi::examples::gui::Control::button(
            "", "Reload current material",
            m_selected_material_supports_reloading
                ? "Reload the module this material is defined in as well as all its dependencies."
                : "Reloading is not supported by this material.",
            m_selected_material_supports_reloading
                ? mi::examples::gui::Control::Flags::None
                : mi::examples::gui::Control::Flags::Disabled))
        {
            create_event(
                static_cast<mi::Size>(Example_dxr_gui_event::Reload_current_material));
        }

        ImGui::Spacing();
    }

    // update all material parameters
    m_internal_section.update(transaction);
    mi::examples::gui::Section_material_update_state state = m_internal_section.get_update_state();

    switch (state)
    {
        // no change
        case mi::examples::gui::Section_material_update_state::No_change:
            break;

        // only class compilation materials changed
        case mi::examples::gui::Section_material_update_state::Argument_block_change:
        {
            // upload changes to the GPU and restart rendering
            m_scene_materials[m_bound_material_index]->update_material_parameters();
            m_scene_data->restart_progressive_rendering();
            // mark changes done
            m_internal_section.reset_update_state();
            break;
        }

        // changes in instance compilation
        // or structural changes in class compilation
        case mi::examples::gui::Section_material_update_state::Structural_change:
        case mi::examples::gui::Section_material_update_state::Unknown_change:
        {
            if(ImGui::IsMouseDown(ImGuiMouseButton_Left))
                break; // wait until the used released the mouse button

            // this will hide the UI in order to avoid further changes while the
            // material is recompiled
            create_event(
                static_cast<mi::Size>(Example_dxr_gui_event::Recompile_current_material));

            // mark changes done
            m_internal_section.reset_update_state();
            break;
        }
    };
}

// ------------------------------------------------------------------------------------------------

void Gui_section_edit_material::init_resource_handling(mdl_d3d12::Mdl_material* material)
{
    m_texture_2ds.clear();
    m_texture_2ds.push_back({ 0, "<invalid>" });

    if (!material)
        return;

    const auto& resource = material->get_resources(Mdl_resource_kind::Texture);
    for (const mdl_d3d12::Mdl_resource_assignment& a : resource)
    {
        if (a.dimension == mdl_d3d12::Texture_dimension::Texture_2D)
            m_texture_2ds.push_back({ a.runtime_resource_id, a.resource_name });
    }
}

// ------------------------------------------------------------------------------------------------

mi::Size Gui_section_edit_material::get_available_resource_count(
    mi::neuraylib::IValue::Kind kind)
{
    switch (kind)
    {
        case mi::neuraylib::IValue::VK_TEXTURE:
            return m_texture_2ds.size();

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            return 1;

        default:
            log_error("[GUI] invalid resource type.", SRC);
            return 1;
    }
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Gui_section_edit_material::get_available_resource_id(
    mi::neuraylib::IValue::Kind kind,
    mi::Size index)
{
    switch (kind)
    {
        case mi::neuraylib::IValue::VK_TEXTURE:
            return index >= m_texture_2ds.size() ? 0 : m_texture_2ds[index].first;

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            return 0;

        default:
            log_error("[GUI] invalid resource type.", SRC);
            return 0;
    }
}

// ------------------------------------------------------------------------------------------------

const char* Gui_section_edit_material::get_available_resource_name(
    mi::neuraylib::IValue::Kind kind,
    mi::Size index)
{
    switch (kind)
    {
        case mi::neuraylib::IValue::VK_TEXTURE:
            return index >= m_texture_2ds.size() ? 0 : m_texture_2ds[index].second.c_str();

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            return "<invalid>";

        default:
            log_error("[GUI] invalid resource type.", SRC);
            return "<invalid>";
        }
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Gui_section_edit_material::get_available_resource_id(
    mi::neuraylib::IValue::Kind kind,
    const char* db_name)
{
    if (!db_name) return -1;

    switch (kind)
    {
        case mi::neuraylib::IValue::VK_TEXTURE:
            for (const auto& pair : m_texture_2ds)
                if (strcmp(pair.second.c_str(), db_name) == 0)
                    return pair.first;
            break;

        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            break;

        default:
            log_error("[GUI] invalid resource type.", SRC);
            break;
    }
    return -1;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Gui_performance_overlay::Gui_performance_overlay(mi::examples::gui::Root* gui)
    : mi::examples::gui::Base_element(gui)
{
}

// ------------------------------------------------------------------------------------------------

void Gui_performance_overlay::update(
    const mdl_d3d12::Update_args& args,
    const Scene_constants& scene_data)
{
    bool show = true;
    ImGui::SetNextWindowPos(ImVec2(35.f, 35.f));
    ImGui::SetNextWindowSize(ImVec2(220, 0));
    ImGui::SetNextWindowBgAlpha(0.5f);
    if (ImGui::Begin("##notitle", &show,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav))
    {
        float number_pos = ImGui::GetContentRegionAvail().x * 0.75f;
        auto print_float_line([&](const char* text, const char* format, float value)
            {
                ImGui::Text(text);
                ImGui::SameLine(number_pos);
                ImGui::Text(format, value);
            });

        ImGui::Text("progressive iteration:");
        ImGui::SameLine(number_pos);
        ImGui::Text("%d", scene_data.progressive_iteration);

        ImGui::Separator();

        print_float_line(       "frame time:", "%.4f", float(args.elapsed_time));
        print_float_line("frames per second:", "%.3f", float(1.0 / args.elapsed_time));
        print_float_line(       "total time:", "%.3f", float(args.total_time));
    }
    ImGui::End();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Info_overlay::Info_overlay(mi::examples::gui::Root* gui)
    : mi::examples::gui::Base_element(gui)
{
}

// ------------------------------------------------------------------------------------------------

void Info_overlay::update(const char* text)
{
    bool show = true;
    size_t width = 270;
    size_t height = 60;

    size_t window_width, window_height;
    get_gui().get_api_interface().get_window_size(window_width, window_height);
    ImGui::SetNextWindowSize(ImVec2(float(width), float(height)));
    ImGui::SetNextWindowPos(ImVec2(
        float (window_width - width) * 0.5f, float(window_height - height) * 0.75f));

    ImGui::SetNextWindowBgAlpha(0.75f);
    if (ImGui::Begin("##notitle", &show,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoNav))
    {
        ImVec2 available = ImGui::GetContentRegionAvail();
        ImVec2 text_size = ImGui::CalcTextSize(text, nullptr, false, available.x);

        ImVec2 pos = ImGui::GetCursorPos();
        pos.x += (available.x - text_size.x) * 0.5f;
        pos.y += (available.y - text_size.y) * 0.5f;

        ImGui::SetCursorPos(pos);
        ImGui::TextWrapped(text);
    }
    ImGui::End();
}

}}}
