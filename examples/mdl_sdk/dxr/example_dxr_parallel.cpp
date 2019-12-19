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

    // when auxiliary buffers are enabled, this index is used to select to one to display
    uint32_t display_buffer_index;
};

enum class Ray_type
{
    Radiance = 0,
    Shadow = 1,

    count,
};

enum class Display_buffer_options
{
    Beauty = 0,
    Albedo,
    Normal,

    count
};

static std::string to_string(Display_buffer_options option)
{
    switch (option)
    {
    case Display_buffer_options::Beauty: return "Beauty";
    case Display_buffer_options::Albedo: return "Albedo";
    case Display_buffer_options::Normal: return "Normal";
    default: return "";
    }
}

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
protected:

    bool initialize(Base_options* options) override
    { 
        // using a shared target code is simpler, one block of shader code is generated for
        // all mdl materials. However, this comes with a price. The compilation and code generation
        // can not be done in parallel. If new materials are loaded into the scene, the entire
        // code block has to be generated again.

        // In this example we use individual targets for each material to overcome this limitation.
        // This also means that setting up the pipeline and handling resource gets more involving.
        options->share_target_code = false;
        //options->force_single_theading = true;

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
        m_currently_reloading = false;

        return true;
    }

    bool load() override
    { 
        Timing t("loading application");
        const Example_dxr_options* options = static_cast<const Example_dxr_options*>(get_options());

        // basic initialization
        // ----------------------------------------------------------------------------------------
        if (options->no_gui)
        {
            m_gui = nullptr;
        }
        else
        {
            m_gui = new Gui(this);  // gui only with a win32 windows
            m_gui->set_reload_material_callback(
                std::bind(&Demo_rtx::reload_material, this, std::placeholders::_1));
        }

        m_show_gui = !options->hide_gui;                    // hide UI on start
        get_window()->set_vsync(false);                     // disable vsync for faster convergence
        m_restart_progressive_rendering_required = true;    // start rendering from scratch

        m_take_screenshot = false;

        // basic resource handling one large descriptor heap (array of different resource views)
        // ----------------------------------------------------------------------------------------
        Descriptor_heap* resource_heap = get_resource_descriptor_heap();

        // create a UAV of output buffer as target texture ray tracing results 
        // and add a UAV to the resource heap
        m_output_buffer = new Texture(
            this, GPU_access::unorder_access, 
            get_window()->get_width(), get_window()->get_height(), 1,
            DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingOutputBuffer");

        m_output_buffer_uav = resource_heap->reserve_views(options->enable_auxiliary ? 4 : 2);
        if (!resource_heap->create_unordered_access_view(m_output_buffer, m_output_buffer_uav))
            return false;

        // the frame buffer uses the same format as the back buffer
        m_frame_buffer = new Texture(
            this, GPU_access::unorder_access, 
            get_window()->get_width(), get_window()->get_height(), 1,
            get_window()->get_back_buffer()->get_format(), "FrameBuffer");

        m_frame_buffer_uav = m_output_buffer_uav.create_offset(1);
        if (!resource_heap->create_unordered_access_view(m_frame_buffer, m_frame_buffer_uav))
            return false;

        // for some post processing effects or for AI denoising, auxiliary outputs are required.
        // from the MDL material perspective albedo (approximation) and normals can be generated.
        if (options->enable_auxiliary)
        {
            m_albedo_buffer = new Texture(
                this, GPU_access::unorder_access,
                get_window()->get_width(), get_window()->get_height(), 1,
                DXGI_FORMAT_R32G32B32A32_FLOAT, "RaytracingAlbedoBuffer");

            m_albedo_buffer_uav = m_output_buffer_uav.create_offset(2);
            if (!resource_heap->create_unordered_access_view(m_albedo_buffer, m_albedo_buffer_uav))
                return false;

            m_normal_buffer = new Texture(
                this, GPU_access::unorder_access,
                get_window()->get_width(), get_window()->get_height(), 1,
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


        // load scene data
        // ----------------------------------------------------------------------------------------
        {
            Timing t("loading scene");

            // two stages, import scene from a specific format
            Loader_gltf loader;

            IScene_loader::Scene_options scene_options;
            scene_options.units_per_meter = options->units_per_meter;
            scene_options.handle_z_axis_up = options->handle_z_axis_up;

            if (!loader.load(options->scene, scene_options)) return false;

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
        m_scene->visit(Scene_node::Kind::Camera, [&](Scene_node* node)
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
            cam_desc.far_plane_distance = 10000.0f;
            cam_desc.name = "Default Camera";

            Transform trafo;
            trafo.translation.y = -3.33f;
            trafo.translation.z = 10.0f;

            m_camera_node = m_scene->create(cam_desc, trafo);

            if (!m_camera_node)
                return false;
        }

        // update aspect ratio
        m_camera_node->get_camera()->set_aspect_ratio(float(get_window()->get_width()) /
                                                      float(get_window()->get_height()));

        // add a CBV for the camera constants to the heap 
        m_camera_cbv = resource_heap->reserve_views(1);
        if (!resource_heap->create_constant_buffer_view(
            m_camera_node->get_camera()->get_constants(), m_camera_cbv))
                return false;

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
        m_scene_constants->data.display_buffer_index = 0;

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

        // add a CBV for the scene constants to the heap 
        m_scene_cbv = resource_heap->reserve_views(1);
        if (!resource_heap->create_constant_buffer_view(m_scene_constants, m_scene_cbv))
            return false;

        // add a SRV of the acceleration data structure to heap
        m_acceleration_structure_srv = resource_heap->reserve_views(1);
        if (!resource_heap->create_shader_resource_view(
            m_scene->get_acceleration_structure(), m_acceleration_structure_srv))
                return false;

        // load environment
        // ----------------------------------------------------------------------------------------
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

        // process materials, generate and compile HLSL Code
        // ----------------------------------------------------------------------------------------
        if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
            return false;

        // commit transaction
        get_mdl_sdk().get_transaction().commit();

        // create ray tracing pipeline, shader binding table
        // ----------------------------------------------------------------------------------------
        if (!update_pipeline())
            return false;
    
        return true;
    }

    void reload_material(Mdl_material* material)
    {
        m_currently_reloading = true;
        auto task = std::thread([&, material]()
        {
            log_info("Started reloading material: " + material->get_name());

            bool targets_changed = false;
            if (!get_mdl_sdk().get_library()->reload_material(material, targets_changed))
            {
                log_error("Reloading material failed.", SRC);
                m_currently_reloading = false;
                return;
            }

            if (!targets_changed)
            {
                log_info("All modules the material depends on were reloaded, but without influence "
                         "on existing material in the scene.");
                m_currently_reloading = false;
                return;
            }

            if (!get_mdl_sdk().get_library()->generate_and_compile_targets())
            {
                log_error("generating and compiling the material library failed.", SRC);
                m_currently_reloading = false;
                return;
            }

            if (!update_pipeline())
            {
                log_error("updating the rendering pipeline failed after reload.", SRC);
                m_currently_reloading = false;
                return;
            }

            log_info("All modules the material depends on were reloaded successfully: " + 
                     material->get_name());
        });
        task.detach();
    }

    bool update_pipeline()
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

        // Compile and libraries (and lists of symbols) to the pipeline 
        // (since this is the only pipeline, ownership is passed too)
        {
            Timing t("compiling non-MDL HLSL");

            std::map<std::string, std::string> defines;
            if (options->enable_auxiliary)
                defines["ENABLE_AUXILIARY"] = std::to_string(1);

            Shader_compiler compiler;
            if (!pipeline->add_library(compiler.compile_shader_library(
                get_executable_folder() + "/content/ray_gen_program.hlsl", &defines), true,
                {"RayGenProgram"}))
                return after_cleanup();

            if (!pipeline->add_library(compiler.compile_shader_library(
                get_executable_folder() + "/content/miss_programs.hlsl"), true,
                {"RadianceMissProgram", "ShadowMissProgram"}))
                return after_cleanup();
        }

        {
            Timing t("setting up ray tracing pipeline");

            // add DXIL libraries compiled for each materials and create individual hitgroups
            // for each material
            if (!mat_library.visit_target_codes([&](Mdl_material_target* target)
            {
                std::string target_code_id = "_" + std::to_string(target->get_id());
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

            // use register(b0,space0) for camera constants
            global_root_signature_dt.register_cbv(0, 0, m_camera_cbv);

            // use register(b1,space0) for scene constants
            global_root_signature_dt.register_cbv(1, 0, m_scene_cbv);

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
                std::string target_code_id = "_" + std::to_string(mat->get_target_code_id());

                // Local root signatures for individual programs
                signature = new Root_signature(this, "ClosestHitGroupSignature" + target_code_id);
                signature->add_flag(D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE);

                // bind vertex buffer to shader register(t1)
                signature->register_srv(1);

                // bind index buffer to shader register(t2)
                signature->register_srv(2);

                // bind index offset to shader register(b2)
                signature->register_constants<uint32_t>(2);

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
                std::string target_code_id = "_" + std::to_string(tc_id);

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
            if (!m_scene->visit(Scene_node::Kind::Mesh, [&](const Scene_node* node)
            {
                const Mesh::Instance* instance = node->get_mesh_instance();
                const Mesh* mesh = instance->get_mesh();

                // set mesh parameters for all parts
                Hit_record_root_arguments local_root_arguments;
                local_root_arguments.vbv_address =
                    mesh->get_vertex_buffer()->get_resource()->GetGPUVirtualAddress();

                local_root_arguments.ibv_address =
                    mesh->get_index_buffer()->get_resource()->GetGPUVirtualAddress();

                // iterate over all mesh parts
                return mesh->visit_geometries([&](const Mesh::Geometry* part)
                {
                    // set parameters per mesh part
                    local_root_arguments.geometry_index_offset = part->get_index_offset();

                    // get material for this part of this instance
                    const IMaterial* material = instance->get_material(part);

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
        Command_queue* command_queue = get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

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

    bool unload() override 
    { 
        if(m_gui) delete m_gui;
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

    // called once per frame before render is started
    void update(const Update_args& args) override 
    {
        // swap pipeline if requested
        if (m_swap_next_frame)
        {
            size_t inactive = m_active_pipeline_index;
            m_active_pipeline_index = m_active_pipeline_index == 0 ? 1 : 0;
            m_swap_next_frame = false;


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

                m_currently_reloading = false;
                m_restart_progressive_rendering_required = true;
            });
            cleanup_thead.detach(); // don't wait for this thread
        }


        // record UI commands
        // ----------------------------------------------------------------------------------------

        // update the UI, which return true in case the rendering has to restarted, 
        // e.g., because of material changes
        if (m_gui)
        {
            m_restart_progressive_rendering_required |= m_gui->update(
                m_scene, args,  m_show_gui && !m_currently_reloading);

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

                // exchange the CBV by the one of the new camera
                get_resource_descriptor_heap()->create_constant_buffer_view(
                    camera->get_constants(), m_camera_cbv);
            }

            // render ui only if not hidden
            if (m_show_gui)
            {
                float w_width = 400.f * get_options()->gui_scale;
                float w_height = float(get_window()->get_height()) - 20.f;
                float w_pos_x = float(get_window()->get_width()) - w_width - 10.f;
                float w_pos_y = 10.f;
                ImGui::SetNextWindowPos(ImVec2(w_pos_x, w_pos_y), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(w_width, w_height), ImGuiCond_FirstUseEver);

                ImGui::Begin("Rendering Settings");

                ImGui::Text("progressive iteration: %d",
                            m_scene_constants->data.progressive_iteration);

                ImGui::Text("frame time: %f", args.elapsed_time);
                ImGui::Text("frame per second: %f", 1.0 / args.elapsed_time);
                ImGui::Text("total time: %f", args.total_time);
                ImGui::Text("paths per second: %d",
                            size_t(double(m_output_buffer->get_width() *
                            m_output_buffer->get_height() *
                            m_scene_constants->data.iterations_per_frame) / args.elapsed_time));

                if (get_options()->enable_auxiliary)
                {
                    ImGui::Dummy(ImVec2(0.0f, 3.0f));
                    ImGui::Text("Display options");
                    ImGui::Separator();

                    std::string current_display_buffer =
                        to_string((Display_buffer_options) m_scene_constants->data.display_buffer_index);
                    if (ImGui::BeginCombo("buffer", current_display_buffer.c_str()))
                    {
                        for (unsigned i = 0; i < (unsigned) Display_buffer_options::count; ++i)
                        {
                            const std::string &name = to_string((Display_buffer_options) i);
                            bool is_selected = (current_display_buffer == name);
                            if (ImGui::Selectable(name.c_str(), is_selected))
                                m_scene_constants->data.display_buffer_index = i;
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                }
                {
                    ImGui::Dummy(ImVec2(0.0f, 3.0f));
                    ImGui::Text("Trace options");
                    ImGui::Separator();

                    if (ImGui::Button("restart progressive rendering"))
                        m_restart_progressive_rendering_required = true;

                    if (ImGui::SliderUint("max path length",
                        &m_scene_constants->data.max_ray_depth, 1, 16))
                        m_restart_progressive_rendering_required = true;

                    bool vsync = get_window()->get_vsync();
                    if (ImGui::Checkbox("enable vsync", &vsync))
                        get_window()->set_vsync(vsync);
                }

                {
                    ImGui::Dummy(ImVec2(0.0f, 3.0f));
                    ImGui::Text("Environment and tone mapping options");
                    ImGui::Separator();

                    if (ImGui::SliderFloat("environment intensity",
                        &m_scene_constants->data.environment_intensity_factor, 0.0f, 2.0f))
                        m_restart_progressive_rendering_required = true;

                    ImGui::SliderFloat("exposure [stops]",
                                       &m_scene_constants->data.exposure_compensation, -3.0f, 3.0f);

                    ImGui::SliderFloat("burnout",
                                       &m_scene_constants->data.burn_out, 0.0f, 1.0f);
                }
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

        // skip rendering while reloading to ensure that resources that are being deleted are not
        // used for rendering. This simplifies the example. This assumes that only the module
        // reload itself can fail, or instances get invalid because of definitions that do not
        // exist anymore, or the list of parameters changed.
        // When an error happens a later stage, during code generation, compiling or while updating
        // the pipeline, the application will end up in undefined state as the resources and the
        // heap might not fit to the old pipeline and binding table.
        if (!m_currently_reloading)
        {
            // resources references in global root signature
            command_list->SetComputeRootSignature(
                m_pipeline[m_active_pipeline_index]->get_global_root_signature()->get_signature());

            ID3D12DescriptorHeap* heaps[] = {get_resource_descriptor_heap()->get_heap()};
            command_list->SetDescriptorHeaps(1, heaps);

            // global root signature
            // - first table: for output buffers, camera, scene constants, acceleration structure
            command_list->SetComputeRootDescriptorTable(
                0, heaps[0]->GetGPUDescriptorHandleForHeapStart());
            // - second table  environment
            command_list->SetComputeRootDescriptorTable(
                1, heaps[0]->GetGPUDescriptorHandleForHeapStart());

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
 
    Texture* m_frame_buffer;
    Descriptor_heap_handle m_frame_buffer_uav;

    Texture* m_output_buffer;
    Descriptor_heap_handle m_output_buffer_uav;

    Texture* m_albedo_buffer;
    Descriptor_heap_handle m_albedo_buffer_uav;

    Texture* m_normal_buffer;
    Descriptor_heap_handle m_normal_buffer_uav;

    Descriptor_heap_handle m_acceleration_structure_srv;
    Raytracing_pipeline* m_pipeline[2];
    Shader_binding_tables* m_shader_binding_table[2];
    size_t m_active_pipeline_index; // pipeline and binding tables can be swapped after updates
    bool m_swap_next_frame;
    bool m_currently_reloading;

    Scene* m_scene;
    Constant_buffer<Scene_constants>* m_scene_constants;
    Descriptor_heap_handle m_scene_cbv;

    Scene_node* m_camera_node;
    Descriptor_heap_handle m_camera_cbv;

    Environment* m_environment;

    Gui* m_gui;
    bool m_show_gui;

    bool m_restart_progressive_rendering_required;
    bool m_take_screenshot;
};


// entry point of the application
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR lpCmdLine, int nCmdShow)
{
    if(AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole())
    {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        SetConsoleOutputCP(CP_UTF8);
    }
    Example_dxr_options options;
    options.window_title = L"MDL Direct3D Raytracing    [Press SPACE to toggle options]";
    options.window_width = 1280;
    options.window_height = 720;

    // parse command line options
    int return_code = 0;
    if (parse_options(options, lpCmdLine, return_code))
    {
        // run the application
        Demo_rtx app;
        return_code = app.run(&options, hInstance, nCmdShow);
    }
    return return_code;
}
