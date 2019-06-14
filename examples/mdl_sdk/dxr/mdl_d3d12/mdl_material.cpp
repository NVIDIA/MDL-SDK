/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_material.h"
#include "base_application.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include "mdl_material_info.h"
#include "texture.h"

#include <iostream>
#include <fstream>

#include "example_shared.h"


namespace mdl_d3d12
{
    Mdl_sdk::Mdl_sdk(Base_application* app)
        : m_app(app)
        , use_class_compilation(app->get_options()->use_class_compilation)
        , m_neuray(nullptr)
        , m_mdl_compiler(nullptr)
        , m_database(nullptr)
        , m_image_api(nullptr)
        , m_hlsl_backend(nullptr)
        , m_global_target(nullptr)
        , m_library(nullptr)
    {

        // Access the MDL SDK
        m_neuray = load_and_get_ineuray();
        if (!m_neuray.is_valid_interface()) {
            log_error("Failed to load the MDL SDK.", SRC);
            return;
        }

        m_mdl_compiler = mi::base::Handle<mi::neuraylib::IMdl_compiler>(
            m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

        // add admin space search paths before user space paths
        auto admin_space_paths = get_mdl_admin_space_search_paths();
        for (const auto& path : admin_space_paths)
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0) 
                log_warning("Failed to add the admin space search path: " + path, SRC);

        auto user_space_paths = get_mdl_user_space_search_paths();
        for (const auto& path : user_space_paths)
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0)
                log_warning("Failed to add the user space search path : " + path, SRC);

        // also add example search paths
        const std::string mdl_root = get_samples_mdl_root();
        if (m_mdl_compiler->add_module_path(mdl_root.c_str()) != 0)
            log_warning("Failed to add the MDL example search path: " + mdl_root, SRC);

        // add search paths
        for (auto path : m_app->get_options()->mdl_paths)
        {
            if (m_mdl_compiler->add_module_path(path.c_str()) != 0) {
                log_error("Failed to add custom search path: " + path, SRC);
                return;
            }
        }

        // Load the FreeImage plugin.
        if (m_mdl_compiler->load_plugin_library("nv_freeimage" MI_BASE_DLL_FILE_EXT) != 0) {
            log_error("Failed to load the 'nv_freeimage' plugin.", SRC);
            return;
        }

        // Start the MDL SDK
        mi::Sint32 result = m_neuray->start();
        if (result != 0) {
            log_error("Failed to start Neuray (MDL SDK) with return code: " + 
                      std::to_string(result), SRC);
            return;
        }

        m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();
        m_image_api = m_neuray->get_api_component<mi::neuraylib::IImage_api>();

        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // create and setup HLSL backend
        m_hlsl_backend = m_mdl_compiler->get_backend(mi::neuraylib::IMdl_compiler::MB_HLSL);

        if (m_hlsl_backend->set_option(
            "num_texture_results", std::to_string(get_num_texture_results()).c_str()))
            return;

        if (m_hlsl_backend->set_option("texture_runtime_with_derivs", "off") != 0)
            return;

        m_context = mdl_factory->create_execution_context();


        // depending on the strategy, materials can be compiled to individually targets, 
        // each using its link unit, or all into one target code, sharing the link unit and 
        // potentially code. Here we go for the second approach.
        m_global_target = new Mdl_target(m_app, this);

        m_library = new Mdl_material_library(m_app, this);
    }

    Mdl_sdk::~Mdl_sdk()
    {
        delete m_library;

        if (m_global_target)
            delete m_global_target;

        m_context = nullptr;
        m_image_api = nullptr;
        m_hlsl_backend = nullptr;
        m_mdl_compiler = nullptr;
        m_database = nullptr;

        // Shut down the MDL SDK
        if (m_neuray->shutdown() != 0) {
            log_error("Failed to shutdown Neuray (MDL SDK).", SRC);
        }
        m_neuray = nullptr;

        if (!unload()) {
            log_error("Failed to unload the MDL SDK.", SRC);
        }
    }


    bool Mdl_sdk::log_messages(const mi::neuraylib::IMdl_execution_context& context)
    {
        std::string last_log;
        for (mi::Size i = 0; i < context.get_messages_count(); ++i)
        {
            last_log.clear();

            mi::base::Handle<const mi::neuraylib::IMessage> message(context.get_message(i));
            last_log += message_kind_to_string(message->get_kind());
            last_log += ": ";
            last_log += message->get_string();

            switch (message->get_severity())
            {
                case mi::base::MESSAGE_SEVERITY_ERROR:
                case mi::base::MESSAGE_SEVERITY_FATAL:
                    log_error(last_log, SRC);
                    break;

                case mi::base::MESSAGE_SEVERITY_WARNING:
                    log_warning(last_log, SRC);
                    break;

                default:
                    log_info(last_log, SRC);
                    break;
            }
        }

        return context.get_error_messages_count() == 0;
    }

    // --------------------------------------------------------------------------------------------
    // helper functions to pass glTF PBR model parameters to the MDL support materials
    
    // pass parameter defined in the scene file to the support material  
    // and return the material name to use
    std::string parameterize_support_material(
       Mdl_sdk& m_sdk, 
       mi::neuraylib::ITransaction& transaction,
       const std::string& scene_directory,
       const IScene_loader::Material& material_desc,
       mi::neuraylib::IExpression_list& parameter_list)
    {
        // create material parameters
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            m_sdk.get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());
        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            mdl_factory->create_value_factory(&transaction));
        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            mdl_factory->create_expression_factory(&transaction));
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            mdl_factory->create_type_factory(&transaction));

        // helper to add a texture if it is available
        auto add_texture = [&](
            const std::string& expression_name,
            const std::string& releative_texture_path, float gamma)
        {
            if (releative_texture_path.empty()) return;

            mi::base::Handle<mi::neuraylib::IImage> image(
                transaction.create<mi::neuraylib::IImage>("Image"));
            std::string image_name = "mdl::" + releative_texture_path + "_image";

            std::string file_path = scene_directory + "/" + releative_texture_path;

            image->reset_file(file_path.c_str());
            transaction.store(image.get(), image_name.c_str());

            mi::base::Handle<mi::neuraylib::ITexture> texture(
                transaction.create<mi::neuraylib::ITexture>("Texture"));
            texture->set_image(image_name.c_str());
            texture->set_gamma(gamma);
            std::string texture_name = "mdl::" + releative_texture_path + "_texture2d";
            transaction.store(texture.get(), texture_name.c_str());

            mi::base::Handle<const mi::neuraylib::IType_texture> type(
                tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
            mi::base::Handle<mi::neuraylib::IValue_texture> value(
                vf->create_texture(type.get(), texture_name.c_str()));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            parameter_list.add_expression(expression_name.c_str(), expr.get());

        };

        // helper to add a color parameter
        auto add_color = [&](
            const std::string& expression_name,
            float r, float g, float b)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_color(r, g, b));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            parameter_list.add_expression(expression_name.c_str(), expr.get());
        };

        // helper to add a float
        auto add_float = [&](
            const std::string& expression_name,
            float x)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_float(x));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            parameter_list.add_expression(expression_name.c_str(), expr.get());
        };

        // helper to add a enum
        auto add_enum = [&](
            const std::string& expression_name,
            mi::Sint32 enum_value)
        {
            mi::base::Handle<const mi::neuraylib::IType_enum> type(
                tf->create_enum("::nvidia::sdk_examples::gltf_support::gltf_alpha_mode"));

            mi::base::Handle<mi::neuraylib::IValue_enum> value(vf->create_enum(type.get()));
            value->set_value(enum_value);
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            parameter_list.add_expression(expression_name.c_str(), expr.get());
        };

        // add the actual parameters to the parameter list
        add_texture("normal_texture", material_desc.normal_texture, 1.0f);
        add_float("normal_scale_factor", material_desc.normal_scale_factor);

        add_texture("occlusion_texture", material_desc.occlusion_texture, 1.0f);
        add_float("occlusion_strength", material_desc.occlusion_strength);

        add_texture("emissive_texture", material_desc.emissive_texture, 2.2f);
        add_color("emissive_factor", material_desc.emissive_factor.x,
            material_desc.emissive_factor.y, material_desc.emissive_factor.z);

        add_enum("alpha_mode", static_cast<mi::Sint32>(material_desc.alpha_mode));
        add_float("alpha_cutoff", material_desc.alpha_cutoff);

        // model dependent parameters
        switch (material_desc.pbr_model)
        {
            case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
            {
                add_texture("diffuse_texture",
                    material_desc.khr_specular_glossiness.diffuse_texture, 2.2f);

                add_color("diffuse_factor",
                    material_desc.khr_specular_glossiness.diffuse_factor.x,
                    material_desc.khr_specular_glossiness.diffuse_factor.y,
                    material_desc.khr_specular_glossiness.diffuse_factor.z);

                add_float("base_alpha",
                    material_desc.khr_specular_glossiness.diffuse_factor.w);

                add_texture("specular_glossiness_texture",
                    material_desc.khr_specular_glossiness.specular_glossiness_texture, 2.2f);

                add_color("specular_factor",
                    material_desc.khr_specular_glossiness.specular_factor.x,
                    material_desc.khr_specular_glossiness.specular_factor.y,
                    material_desc.khr_specular_glossiness.specular_factor.z);
                add_float("glossiness_factor",

                    material_desc.khr_specular_glossiness.glossiness_factor);

                return "gltf_material_khr_specular_glossiness";
            }

            case IScene_loader::Material::Pbr_model::Metallic_roughness:
            default:
            {
                add_texture("base_color_texture", 
                    material_desc.metallic_roughness.base_color_texture, 2.2f);

                add_color("base_color_factor", 
                    material_desc.metallic_roughness.base_color_factor.x,
                    material_desc.metallic_roughness.base_color_factor.y, 
                    material_desc.metallic_roughness.base_color_factor.z);

                add_float("base_alpha", 
                    material_desc.metallic_roughness.base_color_factor.w);

                add_texture("metallic_roughness_texture",
                    material_desc.metallic_roughness.metallic_roughness_texture, 1.0f);

                add_float("metallic_factor", 
                    material_desc.metallic_roughness.metallic_factor);

                add_float("roughness_factor", 
                    material_desc.metallic_roughness.roughness_factor);

                return "gltf_material";
            }
        }
    }


    // --------------------------------------------------------------------------------------------

    Mdl_material_library::Mdl_material_library(Base_application* app, Mdl_sdk* sdk)
        : m_app(app)
        , m_sdk(sdk)
        , m_map_shared()
    {
    }

    Mdl_material_library::~Mdl_material_library()
    {
        for (auto& entry : m_map_shared)
            delete entry.second;
    }

    Mdl_material* Mdl_material_library::create(
        const IScene_loader::Material& material_desc,
        const mi::neuraylib::IExpression_list* parameters)
    {
        Mdl_material* mat = new Mdl_material(m_app, material_desc);
        const std::string& mdl_name = material_desc.name;

        // note, sharing one global target code for that is a design choice
        Mdl_target* target = m_sdk->get_global_target();
        bool first_instance = false;
        mi::base::Handle<mi::neuraylib::IExpression_list> gltf_parameters(nullptr);

        // get transaction from the target we will use
        mi::neuraylib::ITransaction& transaction = m_sdk->get_global_target()->get_transaction();

        // parse the mdl name
        bool is_mdle = mdl_name.rfind(".mdle") != std::string::npos;
        std::string module_name, material_name;
        bool use_gltf_support_material = false;
        if (is_mdle)
        {
            // resolve relative paths within the scene directory
            if(is_absolute_path(mdl_name))
                module_name = mdl_name;
            else
                module_name = m_app->get_options()->scene_directory + "/" + mdl_name;

            material_name = "main";
        }
        else
        {
            // regular MDL name (fully qualified (absolute) mdl material name)
            // [::<package>]::<module>::<material>
            size_t sep_pos = mdl_name.rfind("::");
            if (str_starts_with(mdl_name, "::") && sep_pos != 0)
            {
                module_name = mdl_name.substr(0, sep_pos);
                material_name = mdl_name.substr(sep_pos + 2);
            }
            // handle none mdl materials
            else
            {
                // TODO
                // use a default materials that can deal with infos in the material description of
                // the scene file, e.g. GLTF material parameters

                module_name = "nvidia::sdk_examples::gltf_support";
                use_gltf_support_material = true; 

                // .. but it will be disabled for opaque material instances
                if (material_desc.alpha_mode == IScene_loader::Material::Alpha_mode::Opaque)
                    mat->set_flags(mat->get_flags() | IMaterial::Flags::Opaque);

                if (material_desc.single_sided == true)
                    mat->set_flags(mat->get_flags() | IMaterial::Flags::SingleSided);
            }
        }

        // check if the module and thereby the material definition is already loaded
        const char* module_db_name = m_sdk->get_compiler().get_module_db_name(
            &transaction, module_name.c_str(), &m_sdk->get_context());

        // if not, load it
        if (!module_db_name)
        {
            // load the module that contains the material
            m_sdk->get_compiler().load_module(
                &transaction, module_name.c_str(), &m_sdk->get_context());

            if (!m_sdk->log_messages(m_sdk->get_context())) {
                delete mat;
                return nullptr;
            }

            // get the database name
            module_db_name = m_sdk->get_compiler().get_module_db_name(
                &transaction, module_name.c_str(), &m_sdk->get_context());

            if (!m_sdk->log_messages(m_sdk->get_context()) || !module_db_name)
            {
                delete mat;
                return nullptr;
            }
        }

        // pass parameter defined in the scene file to the support material
        if (use_gltf_support_material)
        {
            mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
                m_sdk->get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());
            mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
                mdl_factory->create_expression_factory(&transaction));

            // create a new parameter list
            gltf_parameters = ef->create_expression_list();

            // set the parameters
            material_name = parameterize_support_material(*m_sdk, transaction,
                m_app->get_options()->scene_directory, material_desc, *gltf_parameters.get());

            // use these parameters as parameters for the created instance
            parameters = gltf_parameters.get();
        }

        // get the loaded material from the database
        std::string material_db_name = std::string(module_db_name) + "::" + material_name;

        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            transaction.access<mi::neuraylib::IMaterial_definition>(material_db_name.c_str()));

        if (!material_definition) {
            log_error("Material '" + material_name + "' not found", SRC);
            delete mat;
            return nullptr;
        }

        // create an material instance (with default parameters, if non are specified)
        mi::Sint32 ret = 0;
        mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
            material_definition->create_material_instance(parameters, &ret));
        if (ret != 0) {
            log_error("Instantiating material '" + material_name + "' failed", SRC);
            delete mat;
            return nullptr;
        }

        // compile the instance
        mi::Uint32 flags = m_sdk->use_class_compilation
            ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
            : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

        mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
            material_instance->create_compiled_material(flags, &m_sdk->get_context()));

        if (!m_sdk->log_messages(m_sdk->get_context())) {
            delete mat;
            return nullptr;
        }


        // check if the material is already present and reuse it in that case
        const mi::base::Uuid hash = compiled_material->get_hash();
        auto found = m_map_shared.find(hash);
        if (found != m_map_shared.end())
        {
            mat->m_shared = found->second;
        }
        else
        {
            // create a new material shared
            mat->m_shared = new Mdl_material_shared();
            first_instance = true;

            // add this material to a link unit

            // since all expression will be in the same link unit, the need to be identified
            std::string scattering_name = "mdl_df_scattering_" + std::to_string(mat->m_material_id);
            std::string opacity_name = "mdl_opacity_" + std::to_string(mat->m_material_id);

            std::string emission_name = "mdl_df_emission_" + std::to_string(mat->m_material_id);
            std::string emission_intensity_name = 
                "mdl_emission_intensity_" + std::to_string(mat->m_material_id);

            std::string thin_walled_name = "mdl_thin_walled_" + std::to_string(mat->m_material_id);

            // select expressions to generate HLSL code for
            std::vector<mi::neuraylib::Target_function_description> selected_functions;

            selected_functions.push_back(mi::neuraylib::Target_function_description(
                "surface.scattering", scattering_name.c_str()));

            selected_functions.push_back(mi::neuraylib::Target_function_description(
                "geometry.cutout_opacity", opacity_name.c_str()));

            selected_functions.push_back(mi::neuraylib::Target_function_description(
                "surface.emission.emission", emission_name.c_str()));

            selected_functions.push_back(mi::neuraylib::Target_function_description(
                "surface.emission.intensity", emission_intensity_name.c_str()));

            selected_functions.push_back(mi::neuraylib::Target_function_description(
                "thin_walled", thin_walled_name.c_str()));

            // add the material to the link unit
            target->get_link_unit().add_material(
                compiled_material.get(),
                selected_functions.data(), selected_functions.size(),
                &m_sdk->get_context());
            if (!m_sdk->log_messages(m_sdk->get_context())) {
                delete mat;
                return nullptr;
            }

            // get the resulting target code information

            // function index for "surface.scattering"
            mat->m_shared->m_constants.scattering_function_index =
                selected_functions[0].function_index == static_cast<mi::Size>(-1)
                ? -1
                : static_cast<int32_t>(selected_functions[0].function_index);

            // function index for "geometry.cutout_opacity"
            mat->m_shared->m_constants.opacity_function_index =
                selected_functions[1].function_index == static_cast<mi::Size>(-1)
                ? -1
                : static_cast<int32_t>(selected_functions[1].function_index);

            // function index for "surface.emission.emission"
            mat->m_shared->m_constants.emission_function_index =
                selected_functions[2].function_index == static_cast<mi::Size>(-1)
                ? -1
                : static_cast<int32_t>(selected_functions[2].function_index);

            // function index for "surface.emission.intensity"
            mat->m_shared->m_constants.emission_intensity_function_index =
                selected_functions[3].function_index == static_cast<mi::Size>(-1)
                ? -1
                : static_cast<int32_t>(selected_functions[3].function_index);

            // function index for "thin_walled"
            mat->m_shared->m_constants.thin_walled_function_index =
                selected_functions[4].function_index == static_cast<mi::Size>(-1)
                ? -1
                : static_cast<int32_t>(selected_functions[4].function_index);

            // constant for the entire material
            mat->m_shared->m_argument_layout_index = selected_functions[0].argument_block_index;

            // keep track of the material
            m_map_shared[hash] = mat->m_shared;
        }

        // add callback to get notified when the target was generated
        // at that point the argument block is handled
        material_definition->retain();
        compiled_material->retain();
        target->add_on_generated(std::bind(&Mdl_material::on_target_code_generated, mat,
            std::placeholders::_1, 
            material_definition.get(), 
            compiled_material.get(), 
            first_instance, 
            std::placeholders::_2));

        return mat;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_target::Resource_callback::Resource_callback(Mdl_target* target)
        : m_target(target)
    {
    }

    mi::Uint32 Mdl_target::Resource_callback::get_resource_index(
        mi::neuraylib::IValue_resource const *resource)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
            m_target->get_target_code());

        mi::Uint32 index = target_code->get_known_resource_index(
            &m_target->get_transaction(), resource);

        if (index > 0)
            return index;

        // invalid (or empty) resource
        const char* name = resource->get_value();
        if (!name)
            return 0;

        m_target->get_resource_names().emplace_back(name);
        return static_cast<mi::Uint32>(m_target->get_resource_names().size() - 1);
    }

    mi::Uint32 Mdl_target::Resource_callback::get_string_index(
        mi::neuraylib::IValue_string const *s)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
            m_target->get_target_code());

        for (mi::Size i=0, n= target_code->get_string_constant_count(); i<n; ++i)
            if (strcmp(target_code->get_string_constant(i), s->get_value()) == 0)
                return static_cast<mi::Uint32>(i);

        log_error("TODO: String constant not found.", SRC);
        return 0;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_target::Mdl_target(Base_application* app, Mdl_sdk* sdk)
        : m_app(app)
        , m_sdk(sdk)
        , m_target_code(nullptr)
        , m_hlsl_source_code("")
        , m_read_only_data_segment(nullptr)
        , m_resource_names()
        , m_resource_callback(new Resource_callback(this))
    {
        mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
        m_transaction = scope->create_transaction();

        m_link_unit = m_sdk->get_backend().create_link_unit(
            m_transaction.get(), &m_sdk->get_context());
    }

    Mdl_target::~Mdl_target()
    {
        m_resource_callback = nullptr;
        m_target_code = nullptr;
        m_link_unit = nullptr;

        if (m_read_only_data_segment) delete m_read_only_data_segment;

        for (auto&& t : m_textures)
            if (t) delete t;

        if (m_transaction)
        {
            m_transaction->commit();
            m_transaction = nullptr;
        }
    }

    const mi::neuraylib::ITarget_code* Mdl_target::get_target_code() const
    {
        if (!m_target_code)
            return nullptr;
            
        m_target_code->retain();
        return m_target_code.get();
    }

    void Mdl_target::add_on_generated(std::function<bool(Mdl_target*, D3DCommandList*)> callback)
    {
        m_on_generated_callbacks.push_back(callback);
    }

    bool Mdl_target::generate()
    {
        if (m_target_code) {
            log_error("Target code was already generated. Call ignored.", SRC);
            return true;
        }

        {
            Timing t("generating target code");
            m_target_code = m_sdk->get_backend().translate_link_unit(
                m_link_unit.get(), &m_sdk->get_context());
        }

        std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        if (!m_sdk->log_messages(m_sdk->get_context()))
        {
            log_error("MDL target code generation failed.", SRC);
            return false;
        }

        Timing t2("loading MDL resources");

        // create a command list for uploading data to the GPU
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();

        // add the invalid texture followed by all textures known to the link unit already
        m_resource_names.emplace_back("");
        for (size_t t = 1, n = m_target_code->get_texture_count(); t < n; ++t)
        {
            if (m_target_code->get_texture_shape(t) !=
                mi::neuraylib::ITarget_code::Texture_shape_2d)
            {
                log_error("Currently, only 2D textures are supported by this example!", SRC);
                return false;
            }

            m_resource_names.emplace_back(m_target_code->get_texture(t));
        }

        // create per material resources, parameter bindings, ...
        // ------------------------------------------------------------
        for (auto&& cb : m_on_generated_callbacks)
            if (!cb(this, command_list))
            {
                log_error("On generate code callback return with failure.", SRC);
                return false;
            }

        // create per target resources
        // --------------------------------------

        // read-only data
        size_t ro_data_seg_index = 0; // ?
        if (m_target_code->get_ro_data_segment_count() > ro_data_seg_index)
        {
            const char* name = m_target_code->get_ro_data_segment_name(ro_data_seg_index);
            m_read_only_data_segment = new Buffer(
                m_app, m_target_code->get_ro_data_segment_size(ro_data_seg_index),
                "MDL_ReadOnly_" + std::string(name));

            m_read_only_data_segment->set_data(
                m_target_code->get_ro_data_segment_name(ro_data_seg_index));
        }
        else
        {
            m_read_only_data_segment = new Buffer(m_app, 4, "MDL_ReadOnly_nullptr");
            uint32_t zero(0);
            m_read_only_data_segment->set_data(&zero);
        }
        Descriptor_heap_handle heap_handle =
            resource_heap.add_shader_resource_view(m_read_only_data_segment, true);
        if (!heap_handle.is_valid()) return false;

        // keep the first handle, this is the offset into the applications global descriptor heap
        m_first_descriptor_heap_gpu_handle = resource_heap.get_gpu_handle(heap_handle);

        // copy data to the GPU
        if (m_read_only_data_segment && !m_read_only_data_segment->upload(command_list))
            return false;

        // Get access to the texture data by the texture database name from the target code.
        // skip the invalid texture that is always present
        for (size_t t = 1, n = m_resource_names.size(); t < n; ++t)
        {
            const char* texture_name = m_resource_names[t].c_str();
            mi::base::Handle<const mi::neuraylib::ITexture> texture(
                m_transaction->access<mi::neuraylib::ITexture>(texture_name));

            mi::base::Handle<const mi::neuraylib::IImage> image(
                m_transaction->access<mi::neuraylib::IImage>(texture->get_image()));

            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());

            float gamma = texture->get_effective_gamma();
            char const* image_type = image->get_type();

            if (image->is_uvtile()) {
                log_error("The example does not support uvtile textures!", SRC);
                return false;
            }

            mi::Uint32 tex_width = canvas->get_resolution_x();
            mi::Uint32 tex_height = canvas->get_resolution_y();
            mi::Uint32 tex_layers = canvas->get_layers_size();

            if (canvas->get_tiles_size_x() != 1 || canvas->get_tiles_size_y() != 1) {
                log_error("The example does not support tiled images!", SRC);
                return false;
            }

            // For simplicity, the texture access functions are only implemented for float4 and
            // gamma is pre-applied here (all images are converted to linear space).

            // Convert to linear color space if necessary
            if (gamma != 1.0f)
            {
                // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
                mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
                    m_sdk->get_image_api().convert(canvas.get(), "Color"));
                gamma_canvas->set_gamma(gamma);
                m_sdk->get_image_api().adjust_gamma(gamma_canvas.get(), 1.0f);
                canvas = gamma_canvas;
            }
            else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
            {
                // Convert to expected format
                canvas = m_sdk->get_image_api().convert(canvas.get(), "Color");
            }

            mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
            mi::Float32 const *tex_data = static_cast<mi::Float32 const *>(tile->get_data());

            // create the d3d texture
            Texture* texture_resource = new Texture(
                m_app, GPU_access::shader_resource, 
                tex_width, 
                tex_height, 
                tex_layers, 
                DXGI_FORMAT_R32G32B32A32_FLOAT, 
                texture_name);

            // copy data to the GPU
            if (!texture_resource->upload(command_list, (const uint8_t*) tex_data)) 
                return false;

            // .. since the compute pipeline is used for ray tracing
            texture_resource->transition_to(
                command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

            // create a resource view on the heap
            Descriptor_heap_handle heap_handle = 
                resource_heap.add_shader_resource_view(texture_resource);
            if (!heap_handle.is_valid()) return false;

            m_textures.push_back(texture_resource);
        }


        // have at least one (in case there is no other invalid, black) texture
        size_t tex_count = std::max(size_t(1), m_textures.size());

        // note that the offset in the heap starts with zero
        // for each target we set 'target_heap_region_start' in the local root signature

        // bind read-only data segment to shader register(t4)
        m_resource_descriptor_table.register_srv(4, 0, 0);

        // bind textures to  shader register(t6) and following
        m_resource_descriptor_table.register_srv(6, 0, 1, tex_count);

        // generate some dxr specific shader code to hook things up
        // -------------------------------------------------------------------

        // generate the actual shader code with the help of some snippets
        m_hlsl_source_code.clear();
        m_hlsl_source_code += "#define MDL_RO_DATA_SEGMENT_SLOT t4\n";
        m_hlsl_source_code += "#define MDL_ARGUMENT_BLOCK_SLOT t5\n";
        m_hlsl_source_code += "#define MDL_TEXTURE_SLOT_BEGIN t6\n";
        m_hlsl_source_code += "#define MDL_TEXTURE_SLOT_COUNT " + std::to_string(tex_count) + "\n";
        m_hlsl_source_code += "#define MDL_TEXTURE_SAMPLER_SLOT s0\n";
        m_hlsl_source_code += "#define MDL_LATLONGMAP_SAMPLER_SLOT s1\n";
        m_hlsl_source_code += "#define MDL_NUM_TEXTURE_RESULTS " + 
                              std::to_string(m_sdk->get_num_texture_results()) + "\n";
        m_hlsl_source_code += "\n";                               
        m_hlsl_source_code += "\n";
        m_hlsl_source_code += "#include \"content/mdl_target_code_types.hlsl\"\n";
        m_hlsl_source_code += "#include \"content/mdl_renderer_runtime.hlsl\"\n\n";
        m_hlsl_source_code += m_target_code->get_code();

        // assuming multiple materials that have been compiled to this target/link unit
        // it has to be possible to select the individual functions based on the hit object


        std::string init_switch_function[2] = {
        {
            "void mdl_bsdf_init(in uint function_index, inout Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        },
        {
            "void mdl_edf_init(in uint function_index, inout Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        }};

        std::string sample_switch_function[2] = {
            "void mdl_bsdf_sample(in uint function_index, inout Bsdf_sample_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_sample(in uint function_index, inout Edf_sample_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string evaluate_switch_function[2] = {
            "void mdl_bsdf_evaluate(in uint function_index, inout Bsdf_evaluate_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_evaluate(in uint function_index, inout Edf_evaluate_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string pdf_switch_function[2] = {
            "void mdl_bsdf_pdf(in uint function_index, inout Bsdf_pdf_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_pdf(in uint function_index, inout Edf_pdf_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n"
        };

        std::string opacity_switch_function =
            "float mdl_geometry_cutout_opacity(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        std::string emission_intensity_switch_function =
            "float3 mdl_emission_intensity(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        std::string thin_walled_switch_function =
            "bool mdl_thin_walled(in uint function_index, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n";

        for (size_t f = 0, n = m_target_code->get_callable_function_count(); f < n; ++f)
        {
            mi::neuraylib::ITarget_code::Function_kind func_kind = 
                m_target_code->get_callable_function_kind(f);

            mi::neuraylib::ITarget_code::Distribution_kind dist_kind = 
                m_target_code->get_callable_function_distribution_kind(f);

            std::string name = m_target_code->get_callable_function(f);
            
            
            if (dist_kind == mi::neuraylib::ITarget_code::DK_NONE)
            {
                if (str_starts_with(name, "mdl_opacity_"))
                {
                    opacity_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
                else if (str_starts_with(name, "mdl_emission_intensity_"))
                {
                    emission_intensity_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
                else if (str_starts_with(name, "mdl_thin_walled_"))
                {
                    thin_walled_switch_function += "       case " + std::to_string(f) + ": " +
                        "return " + name + "(state);\n";
                }
            }
            else if (dist_kind == mi::neuraylib::ITarget_code::DK_BSDF || 
                     dist_kind == mi::neuraylib::ITarget_code::DK_EDF)
            {
                size_t index = static_cast<size_t>(dist_kind) - 1;

                switch (func_kind)
                {
                    case mi::neuraylib::ITarget_code::FK_DF_INIT:
                        init_switch_function[index] += 
                            "       case " + std::to_string(f) + ": " + 
                            name + "(state); return;\n";
                        break;

                    case mi::neuraylib::ITarget_code::FK_DF_SAMPLE:
                        sample_switch_function[index] += 
                            "       case " + std::to_string(f-1) + ": " + 
                            name + "(sret_ptr, state); return;\n";
                        break;

                    case mi::neuraylib::ITarget_code::FK_DF_EVALUATE:
                        evaluate_switch_function[index] += 
                            "       case " + std::to_string(f-2) + ": " + 
                            name + "(sret_ptr, state); return;\n";
                        break;

                    case mi::neuraylib::ITarget_code::FK_DF_PDF:
                        pdf_switch_function[index] += 
                            "       case " + std::to_string(f-3) + ": " + 
                            name + "(sret_ptr, state); return;\n";
                        break;
                }
            }

        }

        for (size_t i = 0; i < 2; ++i)
        {
            init_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += init_switch_function[i];

            sample_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += sample_switch_function[i];

            evaluate_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += evaluate_switch_function[i];

            pdf_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += pdf_switch_function[i];
        }

        opacity_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return 1.0f;\n"
            "}\n\n";
        m_hlsl_source_code += opacity_switch_function;

        emission_intensity_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return float3(1.0f, 1.0f, 1.0f);\n"
            "}\n\n";
        m_hlsl_source_code += emission_intensity_switch_function;

        thin_walled_switch_function +=
            "       default: break;\n"
            "   }\n"
            "   return false;\n"
            "}\n\n";
        m_hlsl_source_code += thin_walled_switch_function;


        // this last snipped contains the actual hit shader and the renderer logic
        // ideally, this is the only part that is handwritten
        m_hlsl_source_code += "\n\n#include \"content/mdl_hit_programs.hlsl\"\n\n";

        // write to file for debugging purpose
        std::ofstream file_stream;
        file_stream.open(get_executable_folder() + "/link_unit_source.hlsl");
        if (file_stream)
        {
            file_stream << m_hlsl_source_code.c_str();
            file_stream.close();
        }

        // provide a dummy implementation for the tint function
        // source += "// dummy implementation due to failed code generation\n"
        //     "float3 mdlcode(Shading_state_material state) { return float3(1, 0, 1); }\n";


        command_queue->execute_command_list(command_list);
        command_queue->flush();

        m_transaction->commit();
        m_transaction = nullptr;

        return true;
    }


    // --------------------------------------------------------------------------------------------
    
    namespace
    {
        static uint32_t material_shared_id_counter = 0;
    }

    Mdl_material_shared::Mdl_material_shared()
        : m_material_shared_id(material_shared_id_counter++)
        , m_argument_layout_index(static_cast<mi::Size>(-1))
    {
    }
    
    // --------------------------------------------------------------------------------------------

    namespace {
        static uint32_t material_id_counter = 0;
    }

    Mdl_material::Mdl_material(Base_application* app, const IScene_loader::Material& material_desc)
        : m_app(app)
        , m_shared(nullptr)
        , m_name(material_desc.name)
        , m_material_id(material_id_counter++)
        , m_flags(IMaterial::Flags::None)
        , m_target(nullptr)
        , m_constants(m_app, m_name + "_Constants")
        , m_argument_block_data(nullptr) // size is not known at this point
        , m_info(nullptr)
    {
        m_constants.data.material_id = m_material_id;

        // create/reserve views
        Descriptor_heap & resource_heap = *m_app->get_resource_descriptor_heap();
        m_constants_cbv = resource_heap.add_constant_buffer_view(&m_constants);
        if (!m_constants_cbv.is_valid()) return;

        // since the arg-block buffer can not be created yet, but a continuous sequence of resource 
        // views is required to work with the binding strategy, a view slot is reserved
        m_argument_block_data_srv = resource_heap.add_empty_view();
        if (!m_argument_block_data_srv.is_valid()) return;

        // keep the first handle, 
        // this is the materials offset into the applications global descriptor heap
        m_material_descriptor_heap_region = resource_heap.get_gpu_handle(m_constants_cbv);
    }


    Mdl_material::~Mdl_material()
    {
        if (m_argument_block_data) delete m_argument_block_data;
        if (m_info) delete m_info;
        
        m_target = nullptr;
        m_shared = nullptr;
    }

    Descriptor_table Mdl_material::s_resource_descriptor_table;
    const Descriptor_table& Mdl_material::get_descriptor_table()
    {
        // note that the offset in the heap starts with zero
        // for each material we set 'material_heap_region_start' in the local root signature
        if (s_resource_descriptor_table.get_size() == 0)
        {
            // bind material constants to shader register(b3)
            s_resource_descriptor_table.register_cbv(3, 0, 0); 

            // bind material argument block to shader register(t5)
            s_resource_descriptor_table.register_srv(5, 0, 1); 
        }
        return s_resource_descriptor_table;
    }

    std::vector<D3D12_STATIC_SAMPLER_DESC> Mdl_material::get_sampler_descriptions()
    {
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;

        // for standard textures
        D3D12_STATIC_SAMPLER_DESC  desc = {};
        //desc.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
        desc.Filter = D3D12_FILTER_ANISOTROPIC;
        desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.MipLODBias = 0.0f;
        desc.MaxAnisotropy = 16;    
        desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        desc.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        desc.MinLOD = 0;
        desc.MaxLOD = 0;  // no mipmaps, otherwise use D3D11_FLOAT32_MAX;
        desc.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        desc.ShaderRegister = 0;                        // bind sampler to shader register(s0)
        desc.RegisterSpace = 0;
        samplers.push_back(desc);

        // for lat long environment maps
        desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        desc.ShaderRegister = 1;                        // bind sampler to shader register(s1)
        samplers.push_back(desc);

        return samplers;
    }

    void Mdl_material::update_material_parameters()
    {
        m_argument_block_data->set_data<char>(m_info->get_argument_block_data());

        // assuming material parameters do not change on a per frame basis ...
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();
        m_argument_block_data->upload(command_list);
        command_queue->execute_command_list(command_list);

        m_constants.data.material_flags = static_cast<uint32_t>(m_flags);
    }


    bool Mdl_material::on_target_code_generated(
        Mdl_target* target,
        const mi::neuraylib::IMaterial_definition* material_definition,
        const mi::neuraylib::ICompiled_material* compiled_material,
        bool first_instance,
        D3DCommandList* command_list)
    {
        m_target = target;

        auto after_clean_up = [&](bool result)
        {
            // drop mdl reference
            material_definition->release();
            compiled_material->release();
            return result;
        };

        // get target code infos for this specific material
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(target->get_target_code());
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
        uint32_t zero(0);


        // if there is already an argument block (material has parameter and class compilation)
        mi::Size layout_index = m_shared->m_argument_layout_index;
        if (layout_index != static_cast<mi::Size>(-1) &&
            layout_index < target_code->get_argument_block_count())
        {
            // get the layout
            mi::base::Handle<const mi::neuraylib::ITarget_value_layout> arg_layout(
                target_code->get_argument_block_layout(layout_index));

            // argument block for class compilation parameter data
            mi::base::Handle<const mi::neuraylib::ITarget_argument_block> arg_block;

            // for the first instances of the materials, the block index is the 
            if (first_instance)
            {
                arg_block = target_code->get_argument_block(layout_index);
            }
            // for further instance, new blocks have to be created
            else
            {
                arg_block = target_code->create_argument_block(
                    layout_index,
                    compiled_material,
                    &target->get_resource_callback());

                if (!arg_block)
                {
                    log_error("Failed to create material argument block: " + m_name, SRC);
                    return after_clean_up(false);
                }
            }

            // create a material info object that allows to change parameters
            m_info = new Mdl_material_info(
                compiled_material,
                material_definition,
                arg_layout.get(),
                arg_block.get());

            m_argument_block_data = new Buffer(m_app, 
                round_to_power_of_two(arg_block->get_size(), 4), m_name + "_ArgumentBlock");
            m_argument_block_data->set_data(arg_block->get_data());
        }

        // if there is no data, a dummy buffer that contains a null pointer is created
        if (!m_argument_block_data)
        {
            m_argument_block_data = new Buffer(m_app, 4, m_name + "_ArgumentBlock_nullptr");
            m_argument_block_data->set_data(&zero);
        }

        // pass the block to the material and create a resource view 
        if (!resource_heap.replace_by_shader_resource_view(
            m_argument_block_data, true, m_argument_block_data_srv))
            return after_clean_up(false);

        // copy data to the GPU
        m_constants.data.shared = m_shared->m_constants; // copy the shared part

        // optimization potential
        m_constants.data.material_flags = static_cast<uint32_t>(m_flags);
        
        m_constants.upload();
        if (!m_argument_block_data->upload(command_list))
            return after_clean_up(false);

        return after_clean_up(true);
    }
}
