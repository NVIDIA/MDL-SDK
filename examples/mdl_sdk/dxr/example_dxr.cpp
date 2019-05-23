#include "mdl_d3d12/mdl_d3d12.h"
#include <wrl.h>
#include <algorithm>
#include <imgui.h>
#include <example_shared.h>
#include "gui.h"
#include "options.h"
using namespace mdl_d3d12;

// constants that are updated once per frame or stay constant
// Note, make sure constant buffer elements are 4x32 bit aligned (important for vectors)
struct Scene_constants
{
    float total_time;
    float delta_time;

    // (progressive) rendering
    uint32_t progressive_iteration;
    uint32_t max_ray_depth;
    uint32_t iterations_per_frame;

    // tone mapping
    float exposure_compensation;
    float firefly_clamp_threshold;
    float burn_out;

    // one additional point light for illustration
    uint32_t point_light_enabled;
    DirectX::XMFLOAT3 point_light_position;
    DirectX::XMFLOAT3 point_light_intensity;

    // environment light
    float environment_intensity_factor;

    // gamma correction while rendering to the frame buffer
    float output_gamma_correction;
};

enum class Ray_type
{
    Radiance = 0,
    Shadow = 1,

    count,
};

// Shader hit record that connects mesh instances and geometry parts with their materials.
struct Hit_record_root_arguments
{
    D3D12_GPU_VIRTUAL_ADDRESS vbv_address;
    D3D12_GPU_VIRTUAL_ADDRESS ibv_address;
    uint32_t geometry_index_offset;
    D3D12_GPU_DESCRIPTOR_HANDLE target_heap_region_start;
    D3D12_GPU_DESCRIPTOR_HANDLE material_heap_region_start;
};

class Demo_rtx : public Base_application
{
public:
    bool load() override
    { 
        Timing t("loading application");
        const Example_dxr_options* options = static_cast<const Example_dxr_options*>(get_options());

        // basic initialization
        // ----------------------------------------------------------------------------------------
        m_gui = options->no_gui ? nullptr : new Gui(this);  // gui only with a win32 windows
        m_show_gui = !options->hide_gui;                    // hide UI on start
        get_window()->set_vsync(false);                     // disable vsync for faster convergence
        m_restart_progressive_rendering_required = true;    // start rendering from scratch

        m_take_screenshot = false;

        // basic resource handling one large descriptor heap (array of different resource views)
        // ----------------------------------------------------------------------------------------

        // create a UAV of output buffer as target texture ray tracing results 
        // and add a UAV to the resource heap
        m_output_buffer = new Texture(
            this, GPU_access::unorder_access, 
            get_window()->get_width(), get_window()->get_height(), 1,
            DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingOutputBuffer");

        m_output_buffer_uav = 
            get_resource_descriptor_heap()->add_unordered_access_view(m_output_buffer);
        if (!m_output_buffer_uav.is_valid()) return false;

        // the frame buffer uses the same format as the back buffer
        m_frame_buffer = new Texture(
            this, GPU_access::unorder_access, 
            get_window()->get_width(), get_window()->get_height(), 1,
            get_window()->get_back_buffer()->get_format(), "FrameBuffer");

        m_frame_buffer_uav = 
            get_resource_descriptor_heap()->add_unordered_access_view(m_frame_buffer);
        if (!m_frame_buffer_uav.is_valid()) return false;

        // load scene data
        // ----------------------------------------------------------------------------------------
        {
            Timing t("loading scene");

            // two stages, import scene from a specific format
            Loader_gltf loader;
            if (!loader.load(options->scene)) return false;

            // replace all materials (for debugging purposes)
            if (!options->user_options.at("override_material").empty())
            {
                std::string qualfied_name = options->user_options.at("override_material");
                if (!str_ends_with(qualfied_name, ".mdle") && !str_starts_with(qualfied_name, "::"))
                    qualfied_name = "::" + qualfied_name;

                loader.replace_all_materials(qualfied_name);
            }

            // then, build the scene graph, with meshes, acceleration structure, ...
            m_scene = new Scene(this, "Scene", static_cast<size_t>(Ray_type::count));
            if (!m_scene->build_scene(loader.get_scene())) return false;
        }

        // get the first camera
        m_camera_node = nullptr;
        m_scene->traverse(Scene_node::Kind::Camera, [&](Scene_node* node)
        {
            m_camera_node = node;
            return false; // abort traversal
        });

        // no camera found
        if (!m_camera_node)
        {
            log_warning("Scene does not contain a camera, therefore a default one is created: " + 
                        options->scene, SRC);

            IScene_loader::Camera cam_desc;
            cam_desc.vertical_fov = mdl_d3d12::PI * 0.25f;
            cam_desc.near_plane_distance = 0.01f;
            cam_desc.far_plane_distance = 1000.0f;
            cam_desc.name = "Default Camera";

            Transform trafo;
            trafo.translation.y = -3.33f;
            trafo.translation.z = 10.0f;

            m_camera_node = m_scene->create(cam_desc, trafo);

            if (!m_camera_node)
                return false;
        }

        // keep the initial pose for resetting it on demand
        m_camera_initial_pose = m_camera_node->get_local_transformation();

        // update aspect ratio
        m_camera_node->get_camera()->set_aspect_ratio(float(get_window()->get_width()) /
                                                      float(get_window()->get_height()));

        // add a CBV for the camera constants to the heap 
        m_camera_cbv = get_resource_descriptor_heap()->add_constant_buffer_view(
            m_camera_node->get_camera()->get_constants());
        if (!m_camera_cbv.is_valid()) return false;

        // same for scene constants
        m_scene_constants = new Constant_buffer<Scene_constants>(this, "SceneConstants");
        m_scene_constants->data.delta_time = 0.0f;
        m_scene_constants->data.total_time = 0.0f;
        m_scene_constants->data.progressive_iteration = 
            static_cast<uint32_t>(options->no_gui ? 1 : options->iterations);
        m_scene_constants->data.max_ray_depth = static_cast<uint32_t>(options->ray_depth);
        m_scene_constants->data.iterations_per_frame = 1;
        m_scene_constants->data.exposure_compensation = 0.0f;
        m_scene_constants->data.burn_out = 0.1f;
        m_scene_constants->data.point_light_enabled = options->point_light_enabled ? 1u : 0u;
        m_scene_constants->data.point_light_position = options->point_light_position;
        m_scene_constants->data.point_light_intensity = options->point_light_intensity;
        m_scene_constants->data.environment_intensity_factor = std::max(0.0f, options->hdr_scale);

        switch (get_window()->get_back_buffer()->get_format())
        {
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
            case DXGI_FORMAT_R32G32B32_FLOAT:
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
                // disable tone mapping for HDR targets
                m_scene_constants->data.output_gamma_correction = 1.0f;
                m_scene_constants->data.burn_out = 1.0f;
                break;

            default:
                m_scene_constants->data.output_gamma_correction = 1.0f / 2.2f;
                break;
        }

        const Descriptor_heap_handle scene_cbv = 
            get_resource_descriptor_heap()->add_constant_buffer_view(m_scene_constants);
        if (!scene_cbv.is_valid()) return false;

        // add a RTV of the acceleration data structure to heap
        const Descriptor_heap_handle acceleration_structure_srv = 
            get_resource_descriptor_heap()->add_shader_resource_view(
                m_scene->get_acceleration_structure());
        if (!acceleration_structure_srv.is_valid()) return false;

        // compile the materials to HLSL
        if (!get_mdl_sdk().get_global_target()->generate()) return false;

        // load environment
        // Note, this uses the neuray and creates a new transaction, 
        // so this can be done after generating the target code (or before creating the target)
        {
            Timing t("loading environment");
            std::string path = get_options()->user_options.at("environment");
            if (!is_absolute_path(path))
                path = options->scene_directory + "/" + path;

            m_environment = new Environment(this, path);

            if (options->firefly_clamp)
            {
                float point_light_itegral = mdl_d3d12::PI * 4.0f *
                    (options->point_light_intensity.x +
                     options->point_light_intensity.y +
                     options->point_light_intensity.z) * 0.333f;

                m_scene_constants->data.firefly_clamp_threshold =
                    std::max(m_environment->get_integral(), point_light_itegral) *
                    4.0f; /* magic number*/
            }
            else
                m_scene_constants->data.firefly_clamp_threshold = -1.0f;

        }

        // create ray tracing pipeline, shader binding table
        // ----------------------------------------------------------------------------------------
        m_pipeline = new Raytracing_pipeline(this, "MainRayTracingPipeline");

        // Compile and libraries (and lists of symbols) to the pipeline 
        // (since this is the only pipeline, ownership is passed too)
        {
            Timing t("compiling HLSL");
            Shader_compiler compiler;
            if (!m_pipeline->add_library(compiler.compile_shader_library(
                get_executable_folder() + "/content/ray_gen_program.hlsl"), true,
                {"RayGenProgram"}))
                return false;

            if (!m_pipeline->add_library(compiler.compile_shader_library(
                get_executable_folder() + "/content/miss_programs.hlsl"), true,
                {"RadianceMissProgram", "ShadowMissProgram"}))
                return false;

            if (!m_pipeline->add_library(compiler.compile_shader_library_from_string(
                get_mdl_sdk().get_global_target()->get_hlsl_source_code(), 
                get_executable_folder() + "/link_unit_source.hlsl"), true,
                {"MdlRadianceClosestHitProgram", "MdlRadianceAnyHitProgram", 
                 "MdlShadowAnyHitProgram"}))
                return false;
        }

        {
            Timing t("setting up ray tracing pipeline");

            // Create and add hit groups to the pipeline.
            // this one will handle the shading of objects with MDL materials
            if (!m_pipeline->add_hitgroup(
                "MdlRadianceHitGroup", 
                "MdlRadianceClosestHitProgram", "MdlRadianceAnyHitProgram", ""))
                    return false;

            // .. this one will deal with shadows cast by objects with MDL materials
            if (!m_pipeline->add_hitgroup(
                "MdlShadowHitGroup", "", "MdlShadowAnyHitProgram", "")) 
                    return false;

            // Global root signature that is applicable to all shader called from dispatch ray
            Descriptor_table global_root_signature_dt;

            // use register(u0,space0) for the output buffer
            global_root_signature_dt.register_uav(0, 0, m_output_buffer_uav);

            // use register(u1,space0) for the frame buffer
            global_root_signature_dt.register_uav(1, 0, m_frame_buffer_uav);

            // use register(b0,space0) for camera constants
            global_root_signature_dt.register_cbv(0, 0, m_camera_cbv);

            // use register(b1,space0) for scene constants
            global_root_signature_dt.register_cbv(1, 0, scene_cbv);

            // use register(t0,space0) for the top-level acceleration structure
            global_root_signature_dt.register_srv(0, 0, acceleration_structure_srv);  
            m_pipeline->get_global_root_signature()->register_dt(global_root_signature_dt);

            // also bind environment resources
            m_pipeline->get_global_root_signature()->register_dt(
                m_environment->get_descriptor_table());

            // MDL uses a small static set of texture samplers 
            auto mdl_samplers = Mdl_material::get_sampler_descriptions();
            for (const auto& s : mdl_samplers)
                m_pipeline->get_global_root_signature()->register_static_sampler(s);

            // Create local root signatures for the individual programs/groups
            Root_signature* signature = new Root_signature(this, "RayGenProgramSignature");
            signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
            if (!signature->finalize()) return false;
            if (!m_pipeline->add_signature_association(signature, true, 
                {"RayGenProgram"})) 
                return false;

            signature = new Root_signature(this, "MissProgramSignature");
            signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);
            if (!signature->finalize()) return false;
            if (!m_pipeline->add_signature_association(signature, true, 
                {"RadianceMissProgram", "ShadowMissProgram"})) 
                return false;

            // Local root signatures for individual programs
            signature = new Root_signature(this, "ClosestHitGroupSignature");
            signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);

            // bind vertex buffer to shader register(t1)
            signature->register_srv(1);

            // bind index buffer to shader register(t2)
            signature->register_srv(2);

            // bind index offset to shader register(b2)
            signature->register_constants<uint32_t>(2);

            // all target level resource are handled by the target
            signature->register_dt(get_mdl_sdk().get_global_target()->get_descriptor_table());

            // all material resource are handled by the material
            signature->register_dt(Mdl_material::get_descriptor_table());
            if (!signature->finalize()) return false;
            if (!m_pipeline->add_signature_association(signature, true, 
                {"MdlRadianceHitGroup"})) return false;

            // since the shadow hit also needs access to the MDL material, at least the
            // 'geometry.cutout_opacity' expression, we simply use the same signature.
            // Without alpha blending or cutout support, an empty signature would be sufficient.
            if (!m_pipeline->add_signature_association(signature, false /* owned by group above*/, 
                {"MdlShadowHitGroup"})) return false;

            // ray tracing settings
            m_pipeline->set_max_payload_size(13 * sizeof(float) + 2 * sizeof(uint32_t));
            m_pipeline->set_max_attribute_size(2 * sizeof(float));

            // we don't use recursion, only direct ray + next event estimation in a loop
            m_pipeline->set_max_recursion_depth(2);

            // complete the setup and make it ready for rendering
            if (!m_pipeline->finalize()) return false;
        }

        // create and fill the binding table to provide resources to the individual programs
        {
            Timing t("creating shader binding table");
            m_shader_binding_table = new Shader_binding_tables(
                m_pipeline,
                m_scene->get_acceleration_structure()->get_ray_type_count(),
                m_scene->get_acceleration_structure()->get_hit_record_count(), 
                "ShaderBindingTable");

            const Shader_binding_tables::Shader_handle raygen_handle = 
                m_shader_binding_table->add_ray_generation_program("RayGenProgram");

            const Shader_binding_tables::Shader_handle miss_handle =
                m_shader_binding_table->add_miss_program(
                    static_cast<size_t>(Ray_type::Radiance), "RadianceMissProgram");

            const Shader_binding_tables::Shader_handle shadow_miss_handle =
                m_shader_binding_table->add_miss_program(
                    static_cast<size_t>(Ray_type::Shadow), "ShadowMissProgram");

            const Shader_binding_tables::Shader_handle hit_handle = 
                m_shader_binding_table->add_hit_group(
                    static_cast<size_t>(Ray_type::Radiance), "MdlRadianceHitGroup");

            const Shader_binding_tables::Shader_handle shadow_hit_handle = 
                m_shader_binding_table->add_hit_group(
                    static_cast<size_t>(Ray_type::Shadow), "MdlShadowHitGroup");


            // iterate over all scene nodes and create local root parameters 
            // for each geometry instance
            if(!m_scene->traverse(Scene_node::Kind::Mesh, [&](const Scene_node* node)
            {
                const Raytracing_acceleration_structure::Instance_handle& instance = 
                    node->get_mesh_instance();

                const Mesh* mesh = node->get_mesh();

                // set mesh parameters for all parts
                Hit_record_root_arguments local_root_arguments;
                local_root_arguments.vbv_address = 
                    mesh->get_vertex_buffer()->get_resource()->GetGPUVirtualAddress();

                local_root_arguments.ibv_address = 
                    mesh->get_index_buffer()->get_resource()->GetGPUVirtualAddress();

                // iterate over all mesh parts
                for (auto part : mesh->get_geometries())
                {
                    // set parameters per mesh part
                    local_root_arguments.geometry_index_offset = part.get_index_offset();

                    // target (link unit) specific resources
                    local_root_arguments.target_heap_region_start = 
                        part.get_material()->get_target_descriptor_heap_region();

                    // material specific resources
                    local_root_arguments.material_heap_region_start = 
                        part.get_material()->get_material_descriptor_heap_region();

                    // index in the shader binding table
                    // compute the hit record index based on ray-type, 
                    // BLAS-instance and geometry index (in BLAS)
                    size_t hit_record_index = 
                        m_scene->get_acceleration_structure()->compute_hit_record_index(
                            static_cast<size_t>(Ray_type::Radiance),
                            instance,
                            part.get_geometry());

                    // set data for this part
                    if (!m_shader_binding_table->set_shader_record(
                        hit_record_index, hit_handle, &local_root_arguments)) 
                            return false;

                    // since shadow ray also need to evaluate the MDL expressions
                    // the same signature and therefore the same record is used
                    hit_record_index =
                        m_scene->get_acceleration_structure()->compute_hit_record_index(
                            static_cast<size_t>(Ray_type::Shadow),
                            instance,
                            part.get_geometry());

                    if (!m_shader_binding_table->set_shader_record(
                        hit_record_index, shadow_hit_handle, &local_root_arguments))
                        return false;
                }

                return true; // continue traversal
            })) return false; // failure in traversal action (returned false)

            // complete the table, no more new elements can be added 
            // (but existing could be changed though /* not implemented */)
            if (!m_shader_binding_table->finalize()) return false;
        }
        // upload the table to the GPU
        Command_queue* command_queue = get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        m_shader_binding_table->upload(command_list);
        m_environment->transition_to(command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        command_queue->execute_command_list(command_list);

        // wait until all tasks are finished
        command_queue->flush();
        return true;
    }

    bool unload() override 
    { 
        if(m_gui) delete m_gui;
        delete m_scene;
        delete m_environment;
        delete m_output_buffer;
        delete m_frame_buffer;
        delete m_pipeline;
        delete m_scene_constants;
        delete m_shader_binding_table;
        return true;
    }

    // called once per frame before render is started
    void update(const Update_args& args) override 
    {
        // record UI commands
        // ----------------------------------------------------------------------------------------

        // update the UI, which return true in case the rendering has to restarted, 
        // e.g., because of material changes
        if (m_gui)
        {
            m_restart_progressive_rendering_required |= m_gui->update(m_scene, args, m_show_gui);

            // in case there are multiple cameras in the scene 
            // and the user activated a different one
            auto selected_camera = m_gui->get_selected_camera();
            if (m_camera_node != selected_camera && selected_camera)
            {
                m_camera_node = selected_camera;
                Camera* camera = m_camera_node->get_camera();

                // update aspect ratio in case the resolution changed
                camera->set_aspect_ratio(float(get_window()->get_width()) /
                                         float(get_window()->get_height()));

                // exchange the CBV to the one of the new camera
                get_resource_descriptor_heap()->replace_by_constant_buffer_view(
                    camera->get_constants(), m_camera_cbv);
            }
      
            // render ui only if not hidden
            if (m_show_gui)
            {
                ImGui::Begin("Rendering Settings", false, ImVec2(400, 350));

                ImGui::Text("progressive iteration: %d", 
                            m_scene_constants->data.progressive_iteration);

                ImGui::Text("iterations per frame: %d", 
                            m_scene_constants->data.iterations_per_frame);

                ImGui::Text("frame time: %f", args.elapsed_time);
                ImGui::Text("total time: %f", args.total_time);
                ImGui::Text("paths per second: %d",
                            size_t(double(m_output_buffer->get_width() * 
                            m_output_buffer->get_height() * 
                            m_scene_constants->data.iterations_per_frame) / args.elapsed_time));

                if (ImGui::SliderUint("max path length", 
                    &m_scene_constants->data.max_ray_depth, 1, 16))
                        m_restart_progressive_rendering_required = true;

                // vsync handling
                bool vsync = get_window()->get_vsync();
                if (ImGui::Checkbox("enable vsync", &vsync))
                    get_window()->set_vsync(vsync);

                if (ImGui::Button("reset camera"))
                    m_camera_node->set_local_transformation(m_camera_initial_pose);

                if (ImGui::Button("restart progressive rendering"))
                    m_restart_progressive_rendering_required = true;

                if (ImGui::SliderFloat("environment intensity",
                        &m_scene_constants->data.environment_intensity_factor, 0.0f, 2.0f))
                            m_restart_progressive_rendering_required = true;

                ImGui::SliderFloat("exposure [stops]",
                                   &m_scene_constants->data.exposure_compensation, -3.0f, 3.0f);

                ImGui::SliderFloat("burnout",
                                   &m_scene_constants->data.burn_out, 0.0f, 1.0f);

                ImGui::End();
            }
        }

        // update scene graph
        // TODO updates of mesh transformations are not applied to the acceleration structure yet
        m_scene->update(args);

        // check if the camera been moved (e.g. by reset)
        if(m_camera_node->transformed_on_last_update())
            m_restart_progressive_rendering_required = true;

        // Update scene constants
        // ----------------------------------------------------------------------------------------
        m_scene_constants->data.delta_time = static_cast<float>(args.elapsed_time);
        m_scene_constants->data.total_time = static_cast<float>(args.total_time);

        // restart progressive rendering
        if (m_restart_progressive_rendering_required)
        {
            m_scene_constants->data.progressive_iteration = 0;
            m_restart_progressive_rendering_required = false;
        }

        m_scene_constants->upload();
    }

    // called once per frame after update is completed
    void render(const Render_args& args) override 
    {
        // get a command list
        Command_queue* command_queue = get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        // bind resources
        // ----------------------------------------------------------------------------------------
        m_frame_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // resources references in global root signature
        command_list->SetComputeRootSignature(
            m_pipeline->get_global_root_signature()->get_signature());

        ID3D12DescriptorHeap* heaps[] = {get_resource_descriptor_heap()->get_heap()};
        command_list->SetDescriptorHeaps(1, heaps);

        command_list->SetComputeRootDescriptorTable(
            0, heaps[0]->GetGPUDescriptorHandleForHeapStart());

        // environment
        command_list->SetComputeRootDescriptorTable(
            1, heaps[0]->GetGPUDescriptorHandleForHeapStart());

        // dispatch rays
        D3D12_DISPATCH_RAYS_DESC desc = m_shader_binding_table->get_dispatch_description();
        desc.Width = static_cast<UINT>(args.back_buffer->get_width());
        desc.Height = static_cast<UINT>(args.back_buffer->get_height());
        command_list->SetPipelineState1(m_pipeline->get_state());
        command_list->DispatchRays(&desc);
        
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
                get_mdl_sdk().get_compiler().export_canvas(
                    get_options()->output_file.c_str(), canvas.get());

            // continue with a new command list
            command_list = command_queue->get_command_list();
        }

        // render UI on top
        if (m_gui)
        {
            args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_RENDER_TARGET);
            command_list->OMSetRenderTargets(1, &args.back_buffer_rtv, FALSE, NULL);
            m_gui->render(command_list, args);
        }

        // indicate that the back buffer will now be used to present.
        args.back_buffer->transition_to(command_list, D3D12_RESOURCE_STATE_PRESENT);
        command_queue->execute_command_list(command_list);
        
        m_scene_constants->data.progressive_iteration += 
            m_scene_constants->data.iterations_per_frame;
    };

    // called on window resize
    void on_resize(size_t width, size_t height) override
    {
        m_output_buffer->resize(width, height);
        get_resource_descriptor_heap()->replace_by_unordered_access_view(
            m_output_buffer, m_output_buffer_uav);

        m_frame_buffer->resize(width, height);
        get_resource_descriptor_heap()->replace_by_unordered_access_view(
            m_frame_buffer, m_frame_buffer_uav);

        if (m_gui)
        {
            m_gui->resize(width, height);
        }

        m_camera_node->get_camera()->set_aspect_ratio(float(width) / float(height));
        m_restart_progressive_rendering_required = true;
    }

    void key_up(uint8_t key) override
    {
        switch (key)
        {
            // close the application
            case VK_ESCAPE:
                get_window()->close();
                break;

            // view/hide ui
            case VK_SPACE:
                m_show_gui = !m_show_gui;
                break;

            // take a screen shot
            case VK_SNAPSHOT:
                m_take_screenshot = true;
                break;

            // toggle full screen
            case VK_F11:
            {
                IWindow::Mode mode = get_window()->get_window_mode();
                switch (mode)
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
                m_restart_progressive_rendering_required = true;
            }
        }
    }


private:
 
    Shader_binding_tables* m_shader_binding_table;

    Texture* m_frame_buffer;
    Descriptor_heap_handle m_frame_buffer_uav;

    Texture* m_output_buffer;
    Descriptor_heap_handle m_output_buffer_uav;

    Raytracing_pipeline* m_pipeline;
    Constant_buffer<Scene_constants>* m_scene_constants;

    Scene* m_scene;

    Scene_node* m_camera_node;
    Descriptor_heap_handle m_camera_cbv;
    Transform m_camera_initial_pose;

    Environment* m_environment;

    Gui* m_gui;
    bool m_show_gui;

    bool m_restart_progressive_rendering_required;
    bool m_take_screenshot;
};


// entry point of the application
_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR command_line_args, int nCmdShow)
{
    if(AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole())
    {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
    }
    Example_dxr_options options;
    options.window_title = "MDL Direct3D Raytracing";
    options.window_width = 1280;
    options.window_height = 720;

    // parse command line options
    int return_code = 0;
    if (parse_options(options, command_line_args, return_code))
    {
        // run the application
        Base_application* app = new Demo_rtx();
        return_code = app->run(&options, hInstance, nCmdShow);
    }
    return return_code;
}
