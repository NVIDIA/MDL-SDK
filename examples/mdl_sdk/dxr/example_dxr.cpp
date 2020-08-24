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

#include "example_dxr.h"

#include "example_dxr_options.h"
#include "example_dxr_gui.h"

#include "mdl_d3d12/mdl_d3d12.h"
#include "mdl_d3d12/camera_controls.h"
#include "mdl_d3d12/mdl_material_description.h"
#include "mdl_d3d12/texture.h"
#include "mdl_d3d12/window_win32.h"

#include <wrl.h>
#include <algorithm>
#include <imgui.h>
#include <gui/gui.h>
#include <gui/gui_api_interface_dx12.h>
#include <gui/gui_material_properties.h>

#include <example_shared.h>

#ifdef MDL_ENABLE_MATERIALX
    #include "mdl_d3d12/materialx/mdl_material_description_loader_mtlx.h"
#endif

namespace mi { namespace examples { namespace dxr
{
using namespace mi::examples::mdl_d3d12;


// Shader hit record that connects mesh instances and geometry parts with their materials.
struct Hit_record_root_arguments
{
    // mesh data
    D3D12_GPU_VIRTUAL_ADDRESS vbv_address;
    D3D12_GPU_VIRTUAL_ADDRESS ibv_address;

    // instance (object) data
    D3D12_GPU_VIRTUAL_ADDRESS instance_scene_data_bv_address;
    D3D12_GPU_VIRTUAL_ADDRESS instance_scene_data_info_bv_address;

    // geometry (mesh part) data
    uint32_t geometry_vertex_buffer_byte_offset;
    uint32_t geometry_vertex_stride;
    uint32_t geometry_index_offset;
    uint32_t geometry_scene_data_info_buffer_offset; // index of the first info

    // material data
    D3D12_GPU_DESCRIPTOR_HANDLE target_heap_region_start;
    D3D12_GPU_DESCRIPTOR_HANDLE material_heap_region_start;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void Scene_constants::restart_progressive_rendering()
{
    progressive_iteration = 0;
}

// ------------------------------------------------------------------------------------------------

void Scene_constants::update_environment(const Environment* environment)
{
    environment_inv_integral = 1.0f / environment->get_integral();
    update_firefly_heuristic(environment);
}

// ------------------------------------------------------------------------------------------------

void Scene_constants::update_firefly_heuristic(const Environment* environment)
{
    if (firefly_clamp_threshold >= 0.0f)
    {
        float point_light_integral = point_light_enabled == 1u
            ? mdl_d3d12::PI * 4.0f * average(point_light_intensity) : 0;

        firefly_clamp_threshold = 4.0f * /* magic number*/
            std::max(
                environment->get_integral() * environment_intensity_factor,
                point_light_integral);
    }
    restart_progressive_rendering();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Example_dxr::Example_dxr()
    : Base_application()
    , m_frame_buffer(nullptr)
    , m_output_buffer(nullptr)
    , m_albedo_buffer(nullptr)
    , m_normal_buffer(nullptr)
    , m_pipeline{ nullptr, nullptr }
    , m_shader_binding_table{ nullptr, nullptr }
    , m_active_pipeline_index(0)
    , m_swap_next_frame(false)
    , m_scene(nullptr)
    , m_scene_constants(nullptr)
    , m_camera_controls(nullptr)
    , m_environment(nullptr)
    , m_take_screenshot(false)
    , m_toggle_fullscreen(false)
    , m_gui_mode(Example_dxr_gui_mode::None)
    , m_main_window_performance_overlay(nullptr)
    , m_info_overlay(nullptr)
{
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::initialize(Base_options* options)
{
    // using a shared target code is simpler, one block of shader code is generated for
    // all mdl materials. However, this comes with a price. The compilation and code generation
    // can not be done in parallel. If new materials are loaded into the scene, the entire
    // code block has to be generated again. Therefore, individual targets for each material
    // are used in this example to overcome this limitation. The shared target code approach
    // can be found in former versions of this example (see repository history).

    //options->force_single_threading = true;

    // when reloading modules and materials, the entire rendering pipeline and the binding
    // tables can get invalid. Therefore the new ones are created after every change. If all
    // updates have been successful, the new pipeline and binding table is swapped with the
    // currently active one for rendering in the next frame.
    m_pipeline[0] = nullptr;
    m_pipeline[1] = nullptr;
    m_shader_binding_table[0] = nullptr;
    m_shader_binding_table[1] = nullptr;
    m_active_pipeline_index = 1;
    m_swap_next_frame = false;

    // hide UI on start
    m_gui_mode = options->hide_gui
        ? Example_dxr_gui_mode::None    // hide GUI by default
        : Example_dxr_gui_mode::All;    // show all we have (for now)

#ifdef MDL_ENABLE_MATERIALX
    // add some MaterialX setup
    // this will probably change with a more sophisticated build integration
    // assuming relevant files have been copied manually into the build folder
    options->mdl_paths.push_back(mi::examples::io::get_executable_folder() + 
        "/autodesk_materialx/source/MaterialXGenMdl/mdl");
#endif

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::load()
{
    Timing t("loading application");
    const Example_dxr_options* options = static_cast<const Example_dxr_options*>(get_options());

    // initialize components that are not available before (window, GPU resources, MDL SDK)
    // ------------------------------------------------------------------------
    get_window()->set_vsync(false);     // disable vsync for faster convergence

    // register code generators
    #ifdef MDL_ENABLE_MATERIALX
        get_mdl_sdk().get_library()->register_mdl_material_description_loader(
            new mi::examples::mdl_d3d12::materialx::Mdl_material_description_loader_mtlx());
    #endif

    // basic resource handling one large descriptor heap (array of different resource views)
    // ------------------------------------------------------------------------
    Descriptor_heap* resource_heap = get_resource_descriptor_heap();

    // create a UAV of output buffer as target texture ray tracing results
    // and add a UAV to the resource heap
    m_output_buffer = Texture::create_texture_2d(
        this, GPU_access::unorder_access,
        get_window()->get_width(), get_window()->get_height(),
        DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingOutputBuffer");

    m_output_buffer_uav = resource_heap->reserve_views(options->enable_auxiliary ? 4 : 2);
    if (!resource_heap->create_unordered_access_view(m_output_buffer, m_output_buffer_uav))
        return false;

    // the frame buffer uses the same format as the back buffer
    m_frame_buffer = Texture::create_texture_2d(
        this, GPU_access::unorder_access,
        get_window()->get_width(), get_window()->get_height(),
        get_window()->get_back_buffer()->get_format(), "FrameBuffer");

    m_frame_buffer_uav = m_output_buffer_uav.create_offset(1);
    if (!resource_heap->create_unordered_access_view(m_frame_buffer, m_frame_buffer_uav))
        return false;

    // for some post processing effects or for AI denoising, auxiliary outputs are required.
    // from the MDL material perspective albedo (approximation) and normals can be generated.
    if (options->enable_auxiliary)
    {
        m_albedo_buffer = Texture::create_texture_2d(
            this, GPU_access::unorder_access,
            get_window()->get_width(), get_window()->get_height(),
            DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingAlbedoBuffer");

        m_albedo_buffer_uav = m_output_buffer_uav.create_offset(2);
        if (!resource_heap->create_unordered_access_view(m_albedo_buffer, m_albedo_buffer_uav))
            return false;

        m_normal_buffer = Texture::create_texture_2d(
            this, GPU_access::unorder_access,
            get_window()->get_width(), get_window()->get_height(),
            DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingNormalBuffer");

        m_normal_buffer_uav = m_output_buffer_uav.create_offset(3);
        if (!resource_heap->create_unordered_access_view(m_normal_buffer, m_normal_buffer_uav))
            return false;
    }
    else
    {
        m_albedo_buffer = nullptr;
        m_normal_buffer = nullptr;
    }

    // create scene constants
    // ------------------------------------------------------------------------
    m_scene_constants = new Dynamic_constant_buffer<Scene_constants>(this, "SceneConstants", 2);
    Scene_constants& scene_data = m_scene_constants->data();
    scene_data.delta_time = 0.0f;
    scene_data.total_time = 0.0f;
    scene_data.progressive_iteration =
        static_cast<uint32_t>(options->no_gui ? 1 : options->iterations);
    scene_data.max_ray_depth = static_cast<uint32_t>(options->ray_depth);
    scene_data.iterations_per_frame = 1;
    scene_data.exposure_compensation = 0.0f;
    scene_data.burn_out = options->tone_mapping_burn_out;
    scene_data.point_light_enabled = options->point_light_enabled ? 1u : 0u;
    scene_data.point_light_position = options->point_light_position;
    scene_data.point_light_intensity = options->point_light_intensity;
    scene_data.environment_intensity_factor = options->hdr_scale;
    scene_data.restart_progressive_rendering();

    if (options->lpe == "albedo")
        scene_data.display_buffer_index = static_cast<unsigned>(Display_buffer_options::Albedo);
    else if (options->lpe == "normal")
        scene_data.display_buffer_index = static_cast<unsigned>(Display_buffer_options::Normal);
    else
        scene_data.display_buffer_index = static_cast<unsigned>(Display_buffer_options::Beauty);

    switch (get_window()->get_back_buffer()->get_format())
    {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            // disable tone mapping for HDR targets
            scene_data.output_gamma_correction = 1.0f;
            scene_data.burn_out = 1.0f;
            break;

        default:
            scene_data.output_gamma_correction = 1.0f / 2.2f;
            break;
    }


    // load environment
    // ------------------------------------------------------------------------
    {
        // if the initial environment map is specified manually (on the command line)
        // the file is resolved in the scene folder, if not available in the working directory.
        std::string env_path = get_options()->user_options.at("environment");
        if (!mi::examples::io::is_absolute_path(env_path))
        {
            std::string p = mi::examples::io::dirname(get_scene_path()) + "/" + env_path;
            if (!mi::examples::io::file_exists(p))
                p = mi::examples::io::get_working_directory() + "/" + env_path;
            env_path = p;
        }

        // use the default texture if the environment does not exist
        if (!mi::examples::io::file_exists(env_path))
            env_path = mi::examples::io::get_executable_folder() +
                "/content/hdri/hdrihaven_teufelsberg_inner_2k.exr";

        // load this environment
        if (!load_environment(env_path, true))
            return false;
    }

    // load scene data
    // ------------------------------------------------------------------------
    {
        // reserve a view for the scenes acceleration data structure
        m_acceleration_structure_srv = resource_heap->reserve_views(1);

        // a camera code is assigned within load_scene
        m_camera_controls = new Camera_controls(this, nullptr);

        // loads the scene and build up all GPU data structures and the pipeline
        if (!load_scene(options->initial_scene, true))
            return false;
    }

    // build the ray tracing pipeline
    // ------------------------------------------------------------------------
    if (!update_rendering_pipeline())
    {
        log_error("updating the rendering pipeline failed after reload.", SRC);
        return false;
    }

    // setup GUI
    // ------------------------------------------------------------------------
    mi::examples::gui::Root* gui = options->no_gui ? nullptr : get_window()->get_gui();
    if (gui)
    {
        // init the GUI system in terms of styles and fonts
        gui->initialize();

        // add sections right panel
        gui->get_panel("right")->add("Rendering", new Gui_section_rendering(
            this, gui, &m_scene_constants->data(), options));

        gui->get_panel("right")->add("Camera", new Gui_section_camera(
            this, gui, &m_scene_constants->data(), m_camera_controls));

        gui->get_panel("right")->add("Light", new Gui_section_light(
            this, gui, &m_scene_constants->data(), options));

        gui->get_panel("right")->add("MDLSettings", new Gui_section_mdl_options(
            this, gui, options));

        gui->get_panel("right")->add("Material", new Gui_section_edit_material(
            this, gui, &m_scene_constants->data()));

        // other UI elements for this window
        m_main_window_performance_overlay = new Gui_performance_overlay(gui);
        m_info_overlay = new Info_overlay(gui);

        // add top menu items
        gui->add(mi::examples::gui::Menu_item("File",
        {
            mi::examples::gui::Menu_item("Open Scene...",
                mi::examples::enums::to_integer(Example_dxr_gui_event::Menu_file_open_scene)),
            mi::examples::gui::Menu_item("Open Environment...",
                mi::examples::enums::to_integer(Example_dxr_gui_event::Menu_file_open_environment)),
            mi::examples::gui::Menu_item("Save Screenshot",
                mi::examples::enums::to_integer(Example_dxr_gui_event::Menu_file_save_screenshot)),
            mi::examples::gui::Menu_item("", mi::examples::gui::Menu_item::Kind::Separator),
            mi::examples::gui::Menu_item("Quit",
                mi::examples::enums::to_integer(Example_dxr_gui_event::Menu_file_quit))
        }));
    }

    set_scene_is_updating(false);
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::load_scene(
    const std::string& scene_path,
    bool skip_pipeline_update)
{
    get_profiling().reset_statistics();
    auto p = get_profiling().measure("loading the scene");

    if (scene_path.empty())
    {
        log_info("No scene selected. Abort loading.");
        return false;
    }

    if (!mi::examples::io::file_exists(scene_path))
    {
        log_error("Scene file '" + scene_path + "' does not exist. Abort loading.");
        return false;
    }

    // skip updates and rendering while reloading the materials
    set_scene_is_updating(true);

    log_info("Started loading scene: " + scene_path);
    const auto options = get_options();

    // two stages, import scene from a specific format
    Loader_gltf loader;

    IScene_loader::Scene_options scene_options;
    scene_options.units_per_meter = options->units_per_meter;
    scene_options.handle_z_axis_up = options->handle_z_axis_up;

    if (!loader.load(scene_path, scene_options))
    {
        log_error("Failed to load scene: " + scene_path);
        set_scene_is_updating(false);
        return false;
    }
    // replace all materials (for debugging purposes)
    std::string override_material;
    if (options->get_user_options("override_material", override_material))
    {
        bool found_replacement = false;

        // check for code generators that match this name
        get_mdl_sdk().get_library()->visit_material_description_loaders(
            [&](const IMdl_material_description_loader* loader)
            {
                if (loader->match_gltf_name(override_material))
                {
                    found_replacement = true;
                    return false; // stop visits
                }
                return true; // continue visits
            });

        // replace with MDL or MDLE material
        std::string mod, mat;
        if (!found_replacement && mi::examples::mdl::parse_cmd_argument_material_name(
            override_material, mod, mat, false))
        {
            override_material = mod + "::" + mat;
            found_replacement = true;
        }

        // use the GLTF support material
        if (!found_replacement)
            log_warning("Using the GLTF Support Material as override because the "
                "provided argument does not match the naming convention for any special "
                "handling like mdl or mdle.");

        loader.replace_all_materials(override_material);
        log_info("All material are replaced by: " + override_material);
    }

    // then, build the scene graph, with meshes, acceleration structure, ...
    set_scene_path(scene_path);
    get_mdl_sdk().reconfigure_search_paths();

    Scene* new_scene = new Scene(this, "Scene", static_cast<size_t>(Ray_type::count));
    if (!new_scene->build_scene(loader.get_scene()))
    {
        log_error("Failed to build scene data structures: " + scene_path);
        set_scene_is_updating(false);
        return false;
    }

    // replace the current with the new one
    Scene* old_scene = m_scene;
    m_scene = new_scene;
    if (old_scene)
        delete old_scene;

    // add a SRV of the acceleration data structure to heap
    if (!get_resource_descriptor_heap()->create_shader_resource_view(
        m_scene->get_acceleration_structure(), m_acceleration_structure_srv))
    {
        set_scene_is_updating(false);
        return false;
    }

    // get the first camera
    bool camera_found = false;
    m_scene->visit(Scene_node::Kind::Camera, [&](Scene_node* node)
    {
        m_camera_controls->set_target(node);
        camera_found = true;

        // update the aspect to match the current window resolution
        node->get_camera()->set_aspect_ratio(
            float(get_window()->get_width()) / float(get_window()->get_height()));

        // scale movement speed with scene size
        const Bounding_box& aabb = m_scene->get_root()->get_global_bounding_box();
        m_camera_controls->movement_speed = length(aabb.size()) / mdl_d3d12::SQRT_3 * 0.5f;
        return false; // abort traversal
    });

    // no camera found
    if (!camera_found)
    {
        log_warning("Scene does not contain a camera, therefore a default one is created: " +
            scene_path);

        IScene_loader::Camera cam_desc;
        cam_desc.vertical_fov = mdl_d3d12::PI * 0.25f;
        cam_desc.near_plane_distance = 0.01f;
        cam_desc.far_plane_distance = 10000.0f;
        cam_desc.name = "Default Camera";
        cam_desc.aspect_ratio =
            float(get_window()->get_width()) / float(get_window()->get_height());

        Transform trafo;
        trafo.translation = { 2.5f, 3.33f, 10.0f };
        trafo.translation *= 10000.f;

        Scene_node* camera_node = m_scene->create(cam_desc, trafo);
        if (!camera_node)
            return false;

        m_camera_controls->set_target(camera_node);

        // fit scene into view
        const Bounding_box& aabb = m_scene->get_root()->get_global_bounding_box();
        m_camera_controls->fit_into_view(aabb);
    }

    // process materials, generate and compile HLSL Code
    // ----------------------------------------------------------------------------------------
    if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
    {
        set_scene_is_updating(false);
        return false;
    }

    // create ray tracing pipeline, shader binding table
    // ----------------------------------------------------------------------------------------
    if (!skip_pipeline_update && !update_rendering_pipeline())
    {
        set_scene_is_updating(false);
        return false;
    }

    // update Gui
    // ----------------------------------------------------------------------------------------
    mi::examples::gui::Root* gui = get_options()->no_gui ? nullptr : get_window()->get_gui();
    if (gui)
    {
        Gui_section_edit_material* mat_gui =
            static_cast<Gui_section_edit_material*>(gui->get_panel("right")->get("Material"));
        if (mat_gui)
        {
            mat_gui->unbind_material();
            mat_gui->update_material_list();
        }
    }

    // continue updates and rendering in the next frame
    m_scene_constants->data().restart_progressive_rendering();
    if (!skip_pipeline_update)
        set_scene_is_updating(false);
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::load_environment(
    const std::string& environment_path,
    bool skip_pipeline_update)
{
    if (environment_path.empty())
    {
        log_info("No environment selected. Abort loading.");
        return false;
    }

    if (!mi::examples::io::file_exists(environment_path))
    {
        log_error("Environment file '" + environment_path + "' does not exist. Abort loading.");
        return false;
    }

    // skip updates and rendering while loading
    set_scene_is_updating(true);

    Timing t("loading environment");
    Environment* new_environment = new Environment(this, environment_path);
    Environment* old_environment = m_environment;
    m_environment = new_environment;
    m_scene_constants->data().update_environment(m_environment);

    // create ray tracing pipeline, shader binding table
    // ----------------------------------------------------------------------------------------
    if (!skip_pipeline_update && !update_rendering_pipeline())
    {
        set_scene_is_updating(false);
        return false;
    }

    m_scene_constants->data().restart_progressive_rendering();
    if (!skip_pipeline_update)
        set_scene_is_updating(false);

    delete old_environment;
    return true;
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::reload_material(
    Mdl_material* material,
    Gui_section_edit_material* mat_gui)
{
    // skip updates and rendering while reloading the materials
    set_scene_is_updating(true);

    std::thread([&, material, mat_gui]()
    {
        // unbind the material from the gui as it can change during the process
        if (mat_gui)
            mat_gui->unbind_material();

        // reload the modules of this material
        log_info("Started reloading material: " + material->get_name());
        bool targets_changed = false;
        if (!get_mdl_sdk().get_library()->reload_material(material, targets_changed))
        {
            log_error("Reloading material failed.", SRC);
            mat_gui->bind_material(material); // rebind the material to the gui
            set_scene_is_updating(false);
            return;
        }

        get_mdl_sdk().get_transaction().commit();

        // in case nothing changed,
        // the update of the targets and the rendering pipeline can be skipped
        if (!targets_changed)
        {
            log_info("All modules the material depends on were reloaded, but without influence "
                "on existing material in the scene.");
            mat_gui->bind_material(material); // rebind the material to the gui
            set_scene_is_updating(false);
            return;
        }

        // if the update fails, continue updates and rendering in the next frame
        // in this case with the previous state
        if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
        {
            log_error("generating and compiling the material library failed.", SRC);
            set_scene_is_updating(false);
            mat_gui->bind_material(material); // rebind the material to the GUI
            return;
        }
        if (!update_rendering_pipeline())
        {
            log_error("updating the rendering pipeline failed after reload.", SRC);
            set_scene_is_updating(false);
            mat_gui->bind_material(material); // rebind the material to the GUI
            return;
        }
        // otherwise, keep in the 'is updating' state until the rendering pipeline is updated, too.

        // reloading was successful at this point
        if (mat_gui)
        {
            // update the material list and rebind the material
            mat_gui->update_material_list();
            mat_gui->bind_material(material);
        }

        log_info("All modules the material depends on were reloaded successfully: " +
            material->get_name());

        // continue updates and rendering in the next frame
        m_scene_constants->data().restart_progressive_rendering();
        set_scene_is_updating(false);

    }).detach();
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::replace_material(
    Mdl_material* material,
    const Mdl_material_description& description,
    Gui_section_edit_material* mat_gui)
{
    // skip updates and rendering while reloading the materials
    set_scene_is_updating(true);

    std::thread([&, material, description, mat_gui]()
    {
        // unbind the material from the GUI as it can change during the process
        if (mat_gui)
            mat_gui->unbind_material();

        // replace the current material
        log_info("Started replacing material: " + material->get_name());
        get_mdl_sdk().get_library()->set_description(material, description);
        get_mdl_sdk().get_transaction().commit();

        // if the update fails, continue updates and rendering in the next frame
        // in this case with the previous state
        if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
        {
            log_error("generating and compiling the material library failed.", SRC);
            set_scene_is_updating(false);
            mat_gui->bind_material(material); // rebind the material to the GUI
            return;
        }
        if (!update_rendering_pipeline())
        {
            log_error("updating the rendering pipeline failed after reload.", SRC);
            set_scene_is_updating(false);
            mat_gui->bind_material(material); // rebind the material to the GUI
            return;
        }
        // otherwise, keep the 'is updating' state until the rendering pipeline is updated.

        // reloading was successful at this point
        if (mat_gui)
        {
            // update the material list and rebind the material
            mat_gui->update_material_list();
            mat_gui->bind_material(material);
        }

        log_info("All modules the material depends on were reloaded successfully: " +
            material->get_name());

        // continue updates and rendering in the next frame
        m_scene_constants->data().restart_progressive_rendering();
        set_scene_is_updating(false);

    }).detach();
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::recompile_materials(
    Mdl_material* selected_material,
    bool recompile_only_the_selected_material,
    Gui_section_edit_material* mat_gui)
{
    // skip updates and rendering while reloading the materials
    set_scene_is_updating(true);

    std::thread([&, selected_material, mat_gui]()
        {
            // unbind the material from the GUI as it can change during the process
            if (mat_gui)
                mat_gui->unbind_material();

            // recompile the material(s)
            if(recompile_only_the_selected_material)
                log_info("Started recompiling material: " + selected_material->get_name());
            else
                log_info("Started recompiling all materials.");

            bool targets_changed = false;
            if (!(recompile_only_the_selected_material
                ? get_mdl_sdk().get_library()->recompile_material(selected_material, targets_changed)
                : get_mdl_sdk().get_library()->recompile_materials(targets_changed)))
            {
                log_error("Recompiling material failed.", SRC);
                mat_gui->bind_material(selected_material); // rebind the material to the GUI
                set_scene_is_updating(false);
                return;
            }

            get_mdl_sdk().get_transaction().commit();

            // in case nothing changed (change of unused parameters)
            // the update of the targets and the rendering pipeline can be skipped
            if (!targets_changed)
            {
                log_info("The material has been recompiled but the changed had no influence.");
                mat_gui->bind_material(selected_material); // rebind the material to the GUI
                set_scene_is_updating(false);
                return;
            }

            // if the update fails, continue updates and rendering in the next frame
            // in this case with the previous state
            if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
            {
                log_error("generating and compiling the material library failed.", SRC);
                set_scene_is_updating(false);
                mat_gui->bind_material(selected_material); // rebind the material to the GUI
                return;
            }
            if (!update_rendering_pipeline())
            {
                log_error("updating the rendering pipeline failed after reload.", SRC);
                set_scene_is_updating(false);
                mat_gui->bind_material(selected_material); // rebind the material to the GUI
                return;
            }
            // otherwise, keep the 'is updating' state until the rendering pipeline is updated.

            // reloading was successful at this point
            if (mat_gui)
            {
                // update the material list and rebind the material
                mat_gui->update_material_list();
                mat_gui->bind_material(selected_material);
            }

            log_info("Material has been recompiled successfully: " +
                selected_material->get_name());

            // continue updates and rendering in the next frame
            m_scene_constants->data().restart_progressive_rendering();
            set_scene_is_updating(false);

        }).detach();
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::update_rendering_pipeline()
{
    const Example_dxr_options* options = static_cast<const Example_dxr_options*>(get_options());
    Mdl_material_library& mat_library = *get_mdl_sdk().get_library();

    Raytracing_pipeline* pipeline = new Raytracing_pipeline(this, "MainRayTracingPipeline");
    Shader_binding_tables* binding_table = nullptr;

    auto after_cleanup = [&]()
    {
        if (pipeline != nullptr)
            delete pipeline;

        if (binding_table != nullptr)
            delete binding_table;

        return false;
    };

    // print debug info
    get_render_target_descriptor_heap()->print_debug_infos();
    get_resource_descriptor_heap()->print_debug_infos();

    // get command list to upload data to the GPU
    Command_queue* command_queue = get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    D3DCommandList* command_list = command_queue->get_command_list();

    // Compile and libraries (and lists of symbols) to the pipeline
    // (since this is the only pipeline, ownership is passed too)
    {
        Timing t("compiling non-MDL HLSL");
        auto p = get_profiling().measure("compiling non-MDL HLSL");

        std::map<std::string, std::string> defines;
        if (options->enable_auxiliary)
            defines["ENABLE_AUXILIARY"] = std::to_string(1);

        Shader_compiler compiler;
        if (!pipeline->add_library(
            compiler.compile_shader_library(
                mi::examples::io::get_executable_folder() + "/content/ray_gen_program.hlsl",
                &defines),
            true,
            {"RayGenProgram"}))
                return after_cleanup();

        if (!pipeline->add_library(
            compiler.compile_shader_library(
                mi::examples::io::get_executable_folder() + "/content/miss_programs.hlsl"),
            true,
            {"RadianceMissProgram", "ShadowMissProgram"}))
                return after_cleanup();
    }

    {
        Timing t("setting up ray tracing pipeline");
        auto p = get_profiling().measure("setting up ray tracing pipeline");

        // add DXIL libraries compiled for each materials and create individual hit-groups
        // for each material
        if (!mat_library.visit_target_codes([&](Mdl_material_target* target)
        {
            std::string target_code_id = "_" + target->get_shader_name_suffix();
            std::string radiance_closest_hit = "MdlRadianceClosestHitProgram" + target_code_id;
            std::string radiance_any_hit = "MdlRadianceAnyHitProgram" + target_code_id;
            std::string shadow_any_hit = "MdlShadowAnyHitProgram" + target_code_id;

            if (!pipeline->add_library(target->get_dxil_compiled_library(), false,
                {radiance_closest_hit, radiance_any_hit, shadow_any_hit}))
                return false;

            // Create and add hit groups to the pipeline.
            // this one will handle the shading of objects with MDL materials
            if (!pipeline->add_hitgroup(
                "MdlRadianceHitGroup" + target_code_id,
                radiance_closest_hit, radiance_any_hit, ""))
                return false;

            // .. this one will deal with shadows cast by objects with MDL materials
            if (!pipeline->add_hitgroup(
                "MdlShadowHitGroup" + target_code_id,
                "", shadow_any_hit, ""))
                return false;

            return true; // continue visits
        })) return after_cleanup();

        // Global root signature that is applicable to all shader called from dispatch ray
        Descriptor_table global_root_signature_dt;

        // use register(u0,space0) for the output buffer
        global_root_signature_dt.register_uav(0, 0, m_output_buffer_uav);

        // use register(u1,space0) for the frame buffer
        global_root_signature_dt.register_uav(1, 0, m_frame_buffer_uav);

        if (options->enable_auxiliary)
        {
            // use register(u2,space0) for the albedo buffer
            global_root_signature_dt.register_uav(2, 0, m_albedo_buffer_uav);

            // use register(u3,space0) for the normal buffer
            global_root_signature_dt.register_uav(3, 0, m_normal_buffer_uav);
        }

        // use register(b0) and (b1) for camera and camera constants
        // place them directly into the root for easier double buffering
        pipeline->get_global_root_signature()->register_cbv(0);
        pipeline->get_global_root_signature()->register_cbv(1);

        // use register(t0,space0) for the top-level acceleration structure
        global_root_signature_dt.register_srv(0, 0, m_acceleration_structure_srv);
        pipeline->get_global_root_signature()->register_dt(global_root_signature_dt);

        // also bind environment resources
        pipeline->get_global_root_signature()->register_dt(
            m_environment->get_descriptor_table());

        // MDL uses a small static set of texture samplers
        auto mdl_samplers = Mdl_material::get_sampler_descriptions();
        for (const auto& s : mdl_samplers)
            pipeline->get_global_root_signature()->register_static_sampler(s);

        // Create local root signatures for the individual programs/groups
        Root_signature* signature = new Root_signature(this, "RayGenProgramSignature");
        signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
        if (!signature->finalize()) return false;
        if (!pipeline->add_signature_association(signature, true,
            {"RayGenProgram"}))
            return false;

        signature = new Root_signature(this, "MissProgramSignature");
        signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
        if (!signature->finalize()) return false;
        if (!pipeline->add_signature_association(signature, true,
            {"RadianceMissProgram", "ShadowMissProgram"}))
            return false;

        // associate the signatures with the hit groups
        if (!mat_library.visit_materials([&](Mdl_material* mat)
        {
            std::string target_code_id = "_" + mat->get_target_code()->get_shader_name_suffix();

            // Local root signatures for individual programs
            signature = new Root_signature(this, "ClosestHitGroupSignature" + target_code_id);
            signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);

            // mesh data
            signature->register_srv(1);                 // vertex buffer to shader register(t1)
            signature->register_srv(2);                 // index buffer to shader register(t2)

            // instance (object) data
            signature->register_srv(3);                 // scene data to register (t3)
            signature->register_srv(4);                 // scene data infos to register (t4)

            // geometry (mesh part) data
            signature->register_constants<uint32_t>(2); // vertex buffer byte offset    (b2)
            signature->register_constants<uint32_t>(3); // vertex stride                (b3)
            signature->register_constants<uint32_t>(4); // index offset                 (b4)
            signature->register_constants<uint32_t>(5); // scene info offset            (b5)

            // all target level resource are handled by the target
            signature->register_dt(mat->get_target_code()->get_descriptor_table());

            // all material resource are handled by the material
            auto material_descriptor_table = mat->get_descriptor_table();
            signature->register_dt(material_descriptor_table);
            if (!signature->finalize()) return false;

            if (!pipeline->add_signature_association(signature, true,
                {"MdlRadianceHitGroup" + target_code_id,
                "MdlRadianceAnyHitProgram" + target_code_id,
                "MdlRadianceClosestHitProgram" + target_code_id})) return false;

            // since the shadow hit also needs access to the MDL material, at least the
            // 'geometry.cutout_opacity' expression, we simply use the same signature.
            // Without alpha blending or cutout support, an empty signature would be sufficient.
            if (!pipeline->add_signature_association(signature, false /*owned by group above*/,
                {"MdlShadowHitGroup" + target_code_id,
                "MdlShadowAnyHitProgram" + target_code_id})) return false;

            return true; // continue visits
        })) return after_cleanup();

        // ray tracing settings
        pipeline->set_max_payload_size(13 * sizeof(float) + 2 * sizeof(uint32_t));
        pipeline->set_max_attribute_size(2 * sizeof(float));

        // we don't use recursion, only direct ray + next event estimation in a loop
        pipeline->set_max_recursion_depth(2);

        // complete the setup and make it ready for rendering
        if (!pipeline->finalize())
            return after_cleanup();
    }

    // create and fill the binding table to provide resources to the individual programs
    {
        Timing t("creating shader binding table");
        auto p = get_profiling().measure("creating shader binding table");
        binding_table = new Shader_binding_tables(
            pipeline,
            m_scene->get_acceleration_structure()->get_ray_type_count(),
            m_scene->get_acceleration_structure()->get_hit_record_count(),
            "ShaderBindingTable");

        const Shader_binding_tables::Shader_handle raygen_handle =
            binding_table->add_ray_generation_program("RayGenProgram");

        const Shader_binding_tables::Shader_handle miss_handle =
            binding_table->add_miss_program(
                static_cast<size_t>(Ray_type::Radiance), "RadianceMissProgram");

        const Shader_binding_tables::Shader_handle shadow_miss_handle =
            binding_table->add_miss_program(
                static_cast<size_t>(Ray_type::Shadow), "ShadowMissProgram");

        std::map<size_t, const Shader_binding_tables::Shader_handle> radiance_hit_handles;
        std::map<size_t, const Shader_binding_tables::Shader_handle> shadow_hit_handles;

        mat_library.visit_target_codes([&](Mdl_material_target* target)
        {
            size_t tc_id = target->get_id();
            std::string target_code_id = "_" + target->get_shader_name_suffix();

            radiance_hit_handles.insert({tc_id, binding_table->add_hit_group(
                static_cast<size_t>(Ray_type::Radiance),
                "MdlRadianceHitGroup" + target_code_id)});

            shadow_hit_handles.insert({tc_id, binding_table->add_hit_group(
                static_cast<size_t>(Ray_type::Shadow),
                "MdlShadowHitGroup" + target_code_id)});

            return true; // continue visit
        });


        // iterate over all scene nodes and create local root parameters
        // for each geometry instance
        if (!m_scene->visit(Scene_node::Kind::Mesh, [&](Scene_node* node)
        {
            Mesh::Instance* instance = node->get_mesh_instance();

            // update scene data of the instance along with per vertex data of the geometry
            // of this instance, it also involves the scene data name that appear in the
            // material assigned to the geometry, so afterwards, material assignments
            // are not possible or require another update of the scene data infos.
            if (!instance->update_scene_data_infos(command_list))
                return false;

            Mesh* mesh = instance->get_mesh();

            // set mesh parameters for all parts
            Hit_record_root_arguments local_root_arguments;

            local_root_arguments.vbv_address =
                mesh->get_vertex_buffer()->get_resource()->GetGPUVirtualAddress();

            local_root_arguments.ibv_address =
                mesh->get_index_buffer()->get_resource()->GetGPUVirtualAddress();

            local_root_arguments.instance_scene_data_bv_address =
                instance->get_scene_data_buffer() ?
                instance->get_scene_data_buffer()->get_resource()->GetGPUVirtualAddress() : 0;

            local_root_arguments.instance_scene_data_info_bv_address =
                instance->get_scene_data_info_buffer()->get_resource()->GetGPUVirtualAddress();

            // iterate over all mesh parts
            return mesh->visit_geometries([&](Mesh::Geometry* part)
            {
                // get material for this part of this instance
                const IMaterial* material = instance->get_material(part);

                // set parameters per mesh part
                local_root_arguments.geometry_vertex_buffer_byte_offset =
                    static_cast<uint32_t>(part->get_vertex_buffer_byte_offset());
                local_root_arguments.geometry_vertex_stride =
                    static_cast<uint32_t>(part->get_vertex_stride());
                local_root_arguments.geometry_index_offset =
                    static_cast<uint32_t>(part->get_index_offset());
                local_root_arguments.geometry_scene_data_info_buffer_offset =
                    static_cast<uint32_t>(part->get_scene_data_info_buffer_offset());

                // get the target code that contains the material
                size_t tc_id = material->get_target_code_id();

                // target (link unit) specific resources
                local_root_arguments.target_heap_region_start =
                    material->get_target_descriptor_heap_region();

                // material specific resources
                local_root_arguments.material_heap_region_start =
                    material->get_material_descriptor_heap_region();

                // index in the shader binding table
                // compute the hit record index based on ray-type,
                // BLAS-instance and geometry index (in BLAS)
                size_t hit_record_index =
                    m_scene->get_acceleration_structure()->compute_hit_record_index(
                        static_cast<size_t>(Ray_type::Radiance),
                        instance->get_instance_handle(),
                        part->get_geometry_handle());

                // set data for this part
                if (!binding_table->set_shader_record(
                    hit_record_index, radiance_hit_handles[tc_id], &local_root_arguments))
                    return false;

                // since shadow ray also need to evaluate the MDL expressions
                // the same signature and therefore the same record is used
                hit_record_index =
                    m_scene->get_acceleration_structure()->compute_hit_record_index(
                        static_cast<size_t>(Ray_type::Shadow),
                        instance->get_instance_handle(),
                        part->get_geometry_handle());

                if (!binding_table->set_shader_record(
                    hit_record_index, shadow_hit_handles[tc_id], &local_root_arguments))
                    return false;

                return true; // continue traversal
            });

        })) return false; // failure in traversal action (returned false)

        // complete the table, no more new elements can be added
        // (but existing could be changed though /* not implemented */)
        if (!binding_table->finalize())
            return after_cleanup();
    }

    // upload the table to the GPU
    binding_table->upload(command_list);
    m_environment->transition_to(command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    command_queue->execute_command_list(command_list);

    // wait until all tasks are finished
    command_queue->flush();


    // set the active one, and swap when updating
    size_t inactive = m_active_pipeline_index == 0 ? 1 : 0;
    m_pipeline[inactive] = pipeline;
    m_shader_binding_table[inactive] = binding_table;
    m_swap_next_frame = true;

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Example_dxr::unload()
{
    if (m_main_window_performance_overlay) delete m_main_window_performance_overlay;
    delete m_camera_controls;
    delete m_scene;
    delete m_environment;
    delete m_output_buffer;
    if (m_albedo_buffer) delete m_albedo_buffer;
    if (m_normal_buffer) delete m_normal_buffer;
    delete m_frame_buffer;
    delete m_scene_constants;
    if (m_pipeline[0]) delete m_pipeline[0];
    if (m_pipeline[1]) delete m_pipeline[1];
    if (m_shader_binding_table[0]) delete m_shader_binding_table[0];
    if (m_shader_binding_table[1]) delete m_shader_binding_table[1];
    return true;
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::update(const Update_args& args)
{
    // swap pipeline after the rendering pipeline was updated due to scene changes
    if (m_swap_next_frame)
    {
        // simply swap pointers
        size_t inactive = m_active_pipeline_index;
        m_active_pipeline_index = m_active_pipeline_index == 0 ? 1 : 0;
        m_swap_next_frame = false;
        m_scene_constants->data().restart_progressive_rendering();

        // free the old data
        auto cleanup_thead = std::thread([&, inactive]()
        {
            if (m_shader_binding_table[inactive] != nullptr)
            {
                delete m_shader_binding_table[inactive];
                m_shader_binding_table[inactive] = nullptr;
            }

            if (m_pipeline[inactive] != nullptr)
            {
                delete m_pipeline[inactive];
                m_pipeline[inactive] = nullptr;
            }
        });

        get_profiling().print_statistics();
        cleanup_thead.detach(); // don't wait for this thread
    }

    // get the GUI instance of the main window
    mi::examples::gui::Root* gui = get_options()->no_gui ? nullptr : get_window()->get_gui();
    if (gui)
        gui->new_frame(); // required even when the main GUI is not rendered

    // skip updates and rendering if the scene changing (geometry, materials, ...)
    if (args.scene_is_updating)
    {
        static const char* info_texts[] =
        {
            "    updating scene    ",
            "    updating scene .  ",
            "    updating scene .. ",
            "    updating scene ..."
        };

        // give some user feedback
        if (gui)
            m_info_overlay->update(info_texts[size_t(args.total_time) % 4]);

        // stop the update pass here
        return;
    }

    // scene constants that are mapped to the GUI and passed into the shader
    Scene_constants& scene_data = m_scene_constants->data();

    // record UI commands
    // ----------------------------------------------------------------------------------------
    if (gui)
    {
        // render main GUI if toggled on
        if (mi::examples::enums::has_flag(m_gui_mode, Example_dxr_gui_mode::Main_gui))
            gui->update(get_mdl_sdk().get_transaction().get());

        // render performance overlay if toggled on
        if (mi::examples::enums::has_flag(m_gui_mode, Example_dxr_gui_mode::Performance_overlay))
            m_main_window_performance_overlay->update(args, m_scene_constants->data());

        // camera controls require ImGui updates
        if (m_camera_controls->update(args))
            m_scene_constants->data().restart_progressive_rendering();

        // process events
        mi::examples::gui::Event e = gui->process_event();
        while (e.is_valid())
        {
            Example_dxr_gui_event event_type_id =
                mi::examples::enums::from_integer<Example_dxr_gui_event>(e.get_event_type_id());

            switch (event_type_id)
            {
                case Example_dxr_gui_event::Reload_current_material:
                {
                    auto mat_gui = static_cast<Gui_section_edit_material*>(e.get_sender());
                    reload_material(mat_gui->get_bound_material(), mat_gui);
                    break;
                }

                case Example_dxr_gui_event::Replace_current_material:
                {
                    auto mat_gui = static_cast<Gui_section_edit_material*>(e.get_sender());
                    mdl_d3d12::Mdl_material* mat = mat_gui->get_bound_material();

                    // keep the GLTF parameters (just in case we can reuse them in the future)
                    IScene_loader::Material parameters =
                        mat->get_material_desciption().get_material_parameters();

                    // only replace the name with the name of the material to load
                    parameters.name = e.get_data<char>();

                    Mdl_material_description desc(parameters);
                    replace_material(mat, desc, mat_gui);
                    break;
                }

                case Example_dxr_gui_event::Recompile_current_material:
                {
                    auto mat_gui = static_cast<Gui_section_edit_material*>(e.get_sender());
                    recompile_materials(mat_gui->get_bound_material(), true, mat_gui);
                    break;
                }

                case Example_dxr_gui_event::Recompile_all_materials:
                {
                    auto mat_gui = static_cast<Gui_section_edit_material*>(
                        get_window()->get_gui()->get_panel("right")->get("Material"));
                    recompile_materials(mat_gui->get_bound_material(), false, mat_gui);
                    break;
                }

                case Example_dxr_gui_event::Menu_file_open_scene:
                    std::thread([&]()
                    {
                        mi::examples::io::open_file_name_dialog dialog;
                        Window_win32* window = static_cast<Window_win32*>(get_window());
                        dialog.set_parent_window(window->get_window_handle());
                        dialog.add_type("GL Transmission Format 2.0", "gltf;glb");
                        std::string scene_path = dialog.show();
                        load_scene(scene_path);
                    }).detach();
                    break;

                case Example_dxr_gui_event::Menu_file_open_environment:
                    std::thread([&]()
                    {
                        mi::examples::io::open_file_name_dialog dialog;
                        Window_win32* window = static_cast<Window_win32*>(get_window());
                        dialog.set_parent_window(window->get_window_handle());
                        dialog.add_type("Environment Map (lat-long)", "hdr;exr");
                        std::string environment_path = dialog.show();
                        load_environment(environment_path);
                    }).detach();
                    break;

                case Example_dxr_gui_event::Menu_file_save_screenshot:
                    m_take_screenshot = true;
                    break;

                case Example_dxr_gui_event::Menu_file_quit:
                    get_window()->close();
                    break;

                default:
                    log_warning("GUI event not handled by the application.");
                    break;
            }
            e = gui->process_event();
        }
    }

    // update scene graph
    // ----------------------------------------------------------------------------------------
    // TODO updates of mesh transformations are not applied to the acceleration structure yet
    if (m_scene->update(args))
        m_scene_constants->data().restart_progressive_rendering();

    // Update scene constants
    // ----------------------------------------------------------------------------------------
    scene_data.delta_time = static_cast<float>(args.elapsed_time);
    scene_data.total_time = static_cast<float>(args.total_time);
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::render(const Render_args& args)
{
    // get a command list
    Command_queue* command_queue = get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    D3DCommandList* command_list = command_queue->get_command_list();

    // bind resources
    // ----------------------------------------------------------------------------------------
    m_frame_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // skip rendering while reloading to ensure that resources that are being deleted are not
    // used for rendering. This simplifies the example. This assumes that only the module
    // reload itself can fail, or instances get invalid because of definitions that do not
    // exist anymore, or the list of parameters changed.
    // When an error happens a later stage, during code generation, compiling or while updating
    // the pipeline, the application will end up in undefined state as the resources and the
    // heap might not fit to the old pipeline and binding table.
    if (!args.scene_is_updating)
    {
        // resources references in global root signature
        command_list->SetComputeRootSignature(
            m_pipeline[m_active_pipeline_index]->get_global_root_signature()->get_signature());

        ID3D12DescriptorHeap* heaps[] = {get_resource_descriptor_heap()->get_heap()};
        command_list->SetDescriptorHeaps(1, heaps);

        // global root signature
        // - direct entries for camera and scene constants
        command_list->SetComputeRootConstantBufferView(0,
            m_camera_controls->get_target()->get_camera()->get_constants()->bind(args));
        command_list->SetComputeRootConstantBufferView(1,
            m_scene_constants->bind(args));

        // - first table: for output buffers, acceleration structure, and materials
        command_list->SetComputeRootDescriptorTable(
            2, heaps[0]->GetGPUDescriptorHandleForHeapStart());
        // - second table  environment
        command_list->SetComputeRootDescriptorTable(
            3, heaps[0]->GetGPUDescriptorHandleForHeapStart());

        // dispatch rays
        D3D12_DISPATCH_RAYS_DESC desc =
            m_shader_binding_table[m_active_pipeline_index]->get_dispatch_description();
        desc.Width = static_cast<UINT>(args.back_buffer->get_width());
        desc.Height = static_cast<UINT>(args.back_buffer->get_height());
        command_list->SetPipelineState1(m_pipeline[m_active_pipeline_index]->get_state());
        command_list->DispatchRays(&desc);
    }

    // copy the ray-tracing buffer to the back buffer
    // ----------------------------------------------------------------------------------------
    m_frame_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_COPY_SOURCE);
    args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_COPY_DEST);
    command_list->CopyResource(
        args.back_buffer->get_resource(), m_frame_buffer->get_resource());

    // write out an image before the UI is added on top
    if (m_take_screenshot)
    {
        m_take_screenshot = false;

        // commit work and make sure the result is ready
        args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_COMMON);
        command_queue->execute_command_list(command_list);
        flush_command_queues();

        // use neuray to write the image file
        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            get_mdl_sdk().get_image_api().create_canvas(
            "Rgba",
            static_cast<mi::Uint32>(args.backbuffer_width),
            static_cast<mi::Uint32>(args.backbuffer_height)));

        mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));

        // download texture and save to output file
        if (args.back_buffer->download(tile->get_data()))
        {
            std::string output_file_name = get_options()->output_file;
            if (!mi::examples::io::is_absolute_path(output_file_name))
                output_file_name = mi::examples::io::get_working_directory() + "/" +
                    output_file_name;

            if (get_mdl_sdk().get_impexp_api().export_canvas(
                output_file_name.c_str(), canvas.get()) != 0)
                    log_error("Failed to save screenshot to: " + output_file_name);
            else
                log_info("Save screenshot to: " + output_file_name);
        }
        // continue with a new command list
        command_list = command_queue->get_command_list();
    }

    // render UI on top
    mi::examples::gui::Root* gui = get_options()->no_gui ? nullptr : get_window()->get_gui();
    if (gui)
    {
        args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_RENDER_TARGET);
        command_list->OMSetRenderTargets(1, &args.back_buffer_rtv, FALSE, NULL);

        mi::examples::gui::Api_interface_dx12::Render_context_dx12 context;
        context.command_list = command_list;
        gui->render(&context);
    }

    // indicate that the back buffer will now be used to present.
    args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_PRESENT);
    command_queue->execute_command_list(command_list);

    Scene_constants& scene_data = m_scene_constants->data();
    scene_data.progressive_iteration +=
        scene_data.iterations_per_frame;

    if (m_toggle_fullscreen)
    {
        m_toggle_fullscreen = false;
        switch (get_window()->get_window_mode())
        {
            case IWindow::Mode::Windowed:
                get_window()->set_window_mode(IWindow::Mode::Fullsceen);
                break;
            case IWindow::Mode::Fullsceen:
                get_window()->set_window_mode(IWindow::Mode::Windowed);
                break;
            default:
                break;
        }
        m_scene_constants->data().restart_progressive_rendering();
    }
};

// ------------------------------------------------------------------------------------------------

void Example_dxr::on_resize(size_t width, size_t height)
{
    m_output_buffer->resize(width, height);
    get_resource_descriptor_heap()->create_unordered_access_view(
        m_output_buffer, m_output_buffer_uav);

    m_frame_buffer->resize(width, height);
    get_resource_descriptor_heap()->create_unordered_access_view(
        m_frame_buffer, m_frame_buffer_uav);

    if (m_albedo_buffer)
    {
        m_albedo_buffer->resize(width, height);
        get_resource_descriptor_heap()->create_unordered_access_view(
            m_albedo_buffer, m_albedo_buffer_uav);

        m_normal_buffer->resize(width, height);
        get_resource_descriptor_heap()->create_unordered_access_view(
            m_normal_buffer, m_normal_buffer_uav);
    }

    m_camera_controls->get_target()->get_camera()->set_aspect_ratio(float(width) / float(height));
    m_scene_constants->data().restart_progressive_rendering();
}

// ------------------------------------------------------------------------------------------------

void Example_dxr::key_up(uint8_t key)
{
    switch (key)
    {
        // close the application
        case VK_ESCAPE:
            get_window()->close();
            break;

        // view/hide GUI
        case VK_SPACE:
            switch (m_gui_mode)
            {
                case Example_dxr_gui_mode::None:
                    m_gui_mode = Example_dxr_gui_mode::All;
                    break;

                case Example_dxr_gui_mode::All:
                    m_gui_mode = Example_dxr_gui_mode::Main_gui;
                    break;

                case Example_dxr_gui_mode::Main_gui:
                    m_gui_mode = Example_dxr_gui_mode::Performance_overlay;
                    break;

                case Example_dxr_gui_mode::Performance_overlay:
                    m_gui_mode = Example_dxr_gui_mode::None;
                    break;

                default:
                    break;
            }
            break;

        // take a screen shot
        case VK_SNAPSHOT:
            m_take_screenshot = true;
            break;

        // toggle full screen
        case VK_F11:
        {
            m_toggle_fullscreen = true;
            break;
        }
    }
}

}}}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// entry point of the application
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR lpCmdLine, int nCmdShow)
{
    // parse command line arguments
    int argc = 0;
    LPWSTR* argv = nullptr;
    if (lpCmdLine && *lpCmdLine)
        argv = CommandLineToArgvW(lpCmdLine, &argc);

    // create a console window of not suppressed by the first argument
    if((argc <= 0 || wcscmp(argv[0], L"--no_console_window") != 0) &&
       (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()))
    {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        SetConsoleOutputCP(CP_UTF8);
    }

    mi::examples::dxr::Example_dxr_options options;
    options.window_title = L"MDL Direct3D Raytracing    [Press SPACE to toggle GUI]";
    options.window_width = 1280;
    options.window_height = 720;

    // parse command line options
    int return_code = 1;
    if (parse_options(options, argc, argv,return_code))
    {
        // run the application
        mi::examples::dxr::Example_dxr app;
        return_code = app.run(&options, hInstance, nCmdShow);
    }
    LocalFree(argv);

    // free logs
    mi::examples::mdl_d3d12::flush_loggers();
    mi::examples::mdl_d3d12::log_set_file_path(nullptr);
    return return_code;
}
