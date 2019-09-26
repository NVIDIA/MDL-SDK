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
        m_mdl_factory = mi::base::Handle<mi::neuraylib::IMdl_factory>(
            m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // create and setup HLSL backend
        m_hlsl_backend = m_mdl_compiler->get_backend(mi::neuraylib::IMdl_compiler::MB_HLSL);

        if (m_hlsl_backend->set_option(
            "num_texture_results", std::to_string(get_num_texture_results()).c_str()))
            return;

        if (m_hlsl_backend->set_option(
            "num_texture_spaces", "2") != 0)
            return;

        if (m_hlsl_backend->set_option("texture_runtime_with_derivs",
            m_app->get_options()->automatic_derivatives ? "on" : "off") != 0)
                return;

        if (m_hlsl_backend->set_option("enable_auxiliary",
            m_app->get_options()->enable_auxiliary ? "on" : "off") != 0)
                return;

        m_transaction = new Mdl_transaction(this);
        m_library = new Mdl_material_library(m_app, this, m_app->get_options()->share_target_code);
    }

    Mdl_sdk::~Mdl_sdk()
    {
        delete m_transaction;
        delete m_library;

        m_mdl_factory = nullptr;
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


    bool Mdl_sdk::log_messages(const mi::neuraylib::IMdl_execution_context* context)
    {
        std::string last_log;
        for (mi::Size i = 0; i < context->get_messages_count(); ++i)
        {
            last_log.clear();

            mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
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

        return context->get_error_messages_count() == 0;
    }

    mi::neuraylib::IMdl_execution_context* Mdl_sdk::create_context()
    {
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            m_mdl_factory->create_execution_context());

        context->set_option("experimental", true);
        context->set_option("fold_ternary_on_df", true);
        context->set_option("internal_space", "coordinate_world");

        context->retain(); // do not free the context right away
        return context.get();
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

            // TODO handle textures that already in the DB because they have been added before

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

            // Mark the texture for removing right away. 
            // Note, this will not delete the data immediately. Instead it will be deleted
            // with the next transaction::commit(). Until then, we will have copied to resources
            // to the GPU.
            transaction.remove(texture_name.c_str());
            transaction.remove(image_name.c_str());

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

    Mdl_material_library::Mdl_material_library(
        Base_application* app, Mdl_sdk* sdk, bool share_target_code)
        : m_app(app)
        , m_sdk(sdk)
        , m_share_target_code(share_target_code)
        , m_target_codes()
        , m_target_codes_mtx()
        , m_map_shared()
        , m_map_shared_mtx()
        , m_materials()
        , m_materials_mtx()
    {
        if (share_target_code)
        {
            // depending on the strategy, materials can be compiled to individually targets, 
            // each using its link unit, or all into one target code, sharing the link unit and 
            // potentially code. Here we go for the second approach.
            m_target_codes[0] = new Mdl_target(m_app, m_sdk);
        }
    }

    Mdl_material_library::~Mdl_material_library()
    {
        for (auto& entry : m_map_shared)
            delete entry.second;

        for (auto& entry : m_target_codes)
            delete entry.second;

        for (auto&& entry : m_textures)
            delete entry.second;

        for (auto&& entry : m_materials)
            delete entry;
    }

    Mdl_target* Mdl_material_library::get_shared_target_code() 
    {
        return m_share_target_code ? m_target_codes[0] : nullptr;
    }

    Mdl_target* Mdl_material_library::get_target_code(size_t target_code_id)
    {
        if(m_share_target_code)
            return nullptr;

        std::lock_guard<std::mutex> lock(m_target_codes_mtx);

        auto it = m_target_codes.find(target_code_id);
        Mdl_target* res = it == m_target_codes.end() ? nullptr : it->second;

        return res;
    }

    Mdl_target* Mdl_material_library::get_target_code_for_material_creation()
    {
        if (m_share_target_code)
            return get_shared_target_code();

        std::lock_guard<std::mutex> lock(m_target_codes_mtx);

        Mdl_target* created = new Mdl_target(m_app, m_sdk);
        m_target_codes[created->get_id()] = created;

        return created;
    }

    Mdl_material* Mdl_material_library::create(
        const IScene_loader::Material& material_desc,
        const mi::neuraylib::IExpression_list* parameters)
    {
        Mdl_material* mat = new Mdl_material(m_app, material_desc);
        const std::string& mdl_name = material_desc.name;

        mi::base::Handle<mi::neuraylib::IExpression_list> gltf_parameters(nullptr);

        // since this method can be called from multiple threads simultaneously
        // a new context for is created
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

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
        const char* module_db_name = m_sdk->get_transaction().execute<const char*>(
            [&](mi::neuraylib::ITransaction* t) 
            {
                return m_sdk->get_compiler().get_module_db_name(
                    t, module_name.c_str(), context.get());
            });

        // if not, load it
        if (!module_db_name)
        {
            // load the module that contains the material
            m_sdk->get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t) 
            {
                // load the module
                m_sdk->get_compiler().load_module(
                    t, module_name.c_str(), context.get());

                // get the database name
                module_db_name = m_sdk->get_compiler().get_module_db_name(
                    t, module_name.c_str(), context.get());

                // Mark the module and all its dependencies for removing right away. 
                // Note, this will not delete the data immediately. Instead it will be deleted
                // with the next transaction::commit(). 
                // Until then, we will have generated the HLSL code.
                //
                // When the application store any objects in the DB, that depends  on our module -- 
                // for example an IMaterial_instance with a complex parameter expression graph 
                // that is worth keeping -- the module (and all its dependencies) are not removed 
                // from the database. That means marking modules for release is safe at this point.
                // However, there is of cause a penalty for future transactions, because modules
                // that have been removed are maybe needed (and loaded) again.
                std::function<void(const char*)> mark_for_removal_recursively = 
                    [&](const char* db_name) 
                    {
                        mi::base::Handle<const mi::neuraylib::IModule> mod(
                            t->access<mi::neuraylib::IModule>(db_name));

                        if (!mod)
                            return;

                        if (mod->is_standard_module()) // keep standard modules
                            return;                    // and maybe a white list for modules you
                                                       // very often. The glTF support module would
                        t->remove(db_name);            // would be a candidate for that.

                        for (mi::Size i = 0, n = mod->get_import_count(); i < n; ++i)
                            mark_for_removal_recursively(mod->get_import(i));
                    };
                mark_for_removal_recursively(module_db_name);
            });

            if (!m_sdk->log_messages(context.get()) || !module_db_name) {
                delete mat;
                return nullptr;
            }
        }

        // pass parameter defined in the scene file to the support material
        if (use_gltf_support_material)
        {
            mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
                m_sdk->get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());

            // create the material parameterization inside a look, since the created textures 
            // a added to the database
            m_sdk->get_transaction().execute<void>(
                [&](mi::neuraylib::ITransaction* t) 
                {
                    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
                        mdl_factory->create_expression_factory(t));

                    // create a new parameter list
                    gltf_parameters = ef->create_expression_list();

                    // set the parameters
                    material_name = parameterize_support_material(
                        *m_sdk, *t,
                        m_app->get_options()->scene_directory, 
                        material_desc, *gltf_parameters.get());

                    // use these parameters as parameters for the created instance
                    parameters = gltf_parameters.get();
                });
        }

        // get the loaded material from the database
        std::string material_db_name = std::string(module_db_name) + "::" + material_name;
        mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
            m_sdk->get_transaction().access<mi::neuraylib::IMaterial_definition>(
                material_db_name.c_str()));

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
            material_instance->create_compiled_material(flags, context.get()));

        if (!m_sdk->log_messages(context.get())) {
            delete mat;
            return nullptr;
        }


        // check if the material is already present and reuse it in that case
        const mi::base::Uuid hash = compiled_material->get_hash();
        bool first_instance = false;
        {
            std::lock_guard<std::mutex> lock(m_map_shared_mtx);

            std::string key = material_db_name + "_";
            key += std::to_string(hash.m_id1);
            key += std::to_string(hash.m_id2);
            key += std::to_string(hash.m_id3);
            key += std::to_string(hash.m_id4);

            auto found = m_map_shared.find(key);
            if (found != m_map_shared.end())
            {
                // simple plausibility check against hash collisions (NOT SAFE)
                if (material_db_name != found->second->m_material_db_name)
                {
                    log_error("Different materials with the same hash detected: \n"
                              "- " + material_db_name + "\n"
                              "- " + found->second->m_material_db_name, SRC);
                }

                // reuse existing material distribution functions and expressions
                mat->m_shared = found->second;
            }
            else
            {
                // create a new material shared
                mat->m_shared = new Mdl_material_shared();
                mat->m_shared->m_material_db_name = material_db_name;

                // get the target code to use
                mat->m_shared->m_target = get_target_code_for_material_creation();

                // keep track of the material 
                m_map_shared[key] = mat->m_shared;

                first_instance = true;
            }
        }

        // add this new material to a link unit
        if(first_instance)
        {
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
            mat->m_shared->m_target->get_link_unit().add_material(
                compiled_material.get(),
                selected_functions.data(), selected_functions.size(),
                context.get());
            if (!m_sdk->log_messages(context.get())) {
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

            // get the maximum number of texture slots of this material
            // and thereby the minimum number of texture resource slots for the target code
            for (size_t a = 0, n = compiled_material->get_parameter_count(); a < n; ++a)
            {
                mi::base::Handle<const mi::neuraylib::IValue> v(compiled_material->get_argument(a));
                switch (v->get_kind())
                {
                    case mi::neuraylib::IValue::VK_TEXTURE:
                        mat->m_shared->m_max_texture_count++;
                    default:
                        break;
                }
            }

            // per material textures are not required in shared target code mode
            // this is stored to set 'MDL_MATERIAL_TEXTURE_SLOT_COUNT' constant in the HLSL code
            if(!m_app->get_options()->share_target_code)
                mat->m_shared->m_target->update_min_texture_count(
                    mat->m_shared->m_max_texture_count);
        }

        // add callback to get notified when the target was generated
        // at that point the argument block is handled
        material_definition->retain();
        compiled_material->retain();
        mat->m_shared->m_target->add_on_generated(std::bind(&Mdl_material::on_target_code_generated, 
            mat,
            material_definition.get(), 
            compiled_material.get(), 
            first_instance, 
            std::placeholders::_1));

        std::lock_guard<std::mutex> lock(m_materials_mtx);
        m_materials.push_back(mat);
        return mat;
    }

    bool Mdl_material_library::visit_target_codes(std::function<bool(Mdl_target*)> action)
    {
        bool success = true;
        std::lock_guard<std::mutex> lock(m_target_codes_mtx);
        for (auto it = m_target_codes.begin(); it != m_target_codes.end(); it++)
        {
            success &= action(it->second);
            if (!success) 
                break;
        }
        return success;
    }

    bool Mdl_material_library::visit_materials(std::function<bool(Mdl_material*)> action)
    {
        bool success = true;
        std::lock_guard<std::mutex> lock(m_materials_mtx);
        for (auto it = m_materials.begin(); it != m_materials.end(); it++)
        {
            success &= action(*it);
            if (!success)
                break;
        }
        return success;
    }

    Texture* Mdl_material_library::access_texture_resource(
        std::string db_name, D3DCommandList* command_list)
    {
        // check if the texture already exists
        {
            std::lock_guard<std::mutex> lock(m_textures_mtx);
            auto found = m_textures.find(db_name);
            if (found != m_textures.end())
                return found->second;
        }

        // if not, collect all infos required to create the texture
        mi::base::Handle<const mi::neuraylib::ITexture> texture(
            m_sdk->get_transaction().access<mi::neuraylib::ITexture>(db_name.c_str()));

        mi::base::Handle<const mi::neuraylib::IImage> image(
            m_sdk->get_transaction().access<mi::neuraylib::IImage>(texture->get_image()));

        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());

        float gamma = texture->get_effective_gamma();
        char const* image_type = image->get_type();

        if (image->is_uvtile()) {
            log_error("The example does not support uvtile textures!", SRC);
            return nullptr;
        }

        mi::Uint32 tex_width = canvas->get_resolution_x();
        mi::Uint32 tex_height = canvas->get_resolution_y();
        mi::Uint32 tex_layers = canvas->get_layers_size();

        if (canvas->get_tiles_size_x() != 1 || canvas->get_tiles_size_y() != 1) {
            log_error("The example does not support tiled images!", SRC);
            return nullptr;
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
            db_name);

        // at this point we are pretty sure that the resource is available and valid
        // so we add it, unless another thread was faster, in which case we discard ours
        {
            std::lock_guard<std::mutex> lock(m_textures_mtx);
            auto found = m_textures.find(db_name);
            if (found != m_textures.end())
            {
                delete texture_resource; // delete ours (before we actually copy data to the GPU)
                return found->second; // return the other
            }

            // add the created texture to the library
            m_textures[db_name] = texture_resource;
        }

        // copy data to the GPU
        if (!texture_resource->upload(command_list, (const uint8_t*)tex_data))
            return nullptr;

        // .. since the compute pipeline is used for ray tracing
        texture_resource->transition_to(
            command_list, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        return texture_resource;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_target::Resource_callback::Resource_callback(
        Mdl_sdk* sdk, Mdl_target* target, Mdl_material* material)
        : m_sdk(sdk)
        , m_target(target)
        , m_material(material)
    {
    }

    mi::Uint32 Mdl_target::Resource_callback::get_resource_index(
        mi::neuraylib::IValue_resource const *resource)
    {
        // resource available in the target code.
        // this is the case for resources that are in the material body and for
        // resources in contained in the parameters of the first appearance of a material
        mi::Uint32 index = m_sdk->get_transaction().execute<mi::Uint32>(
            [&](mi::neuraylib::ITransaction* t) {
                mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                    m_target->get_target_code());
                return target_code->get_known_resource_index(t, resource);
            });
        if (index > 0)
            return index;
        
        // invalid (or empty) resource
        const char* name = resource->get_value();
        if (!name)
        {
            return 0;
        }

        // All resources that are loaded for later appearances of a material, i.e. when a
        // material is reused (probably with different parameters), have to handled separately.
        // If the target was not yet generated (usually the case when a shared target code is used),
        // additional resources can be added to the list of resources of the target.
        // Otherwise, resources are added to the material (when separate link units are used).

        // store textures at the material (when using no shared target)
        if (m_material != nullptr)
        {
            m_material->get_resource_names().emplace_back(name);
            index = static_cast<mi::Uint32>(
                (m_material->get_resource_names().size() - 1) +
                (m_target->get_resource_names().size() - 1));
        }
        // otherwise store it at target code level
        else
        {
            m_target->get_resource_names().emplace_back(name);
            index = static_cast<mi::Uint32>(m_target->get_resource_names().size() - 1);
        }
        
        // log these manually defined indices
        log_info(
            "target code id: " + std::to_string(m_target->get_id()) +
            " - texture id: " + std::to_string(index) +
            (m_material ? (" (material id: " + std::to_string(m_material->get_id()) + ")") : "") +
            " - resource: " + std::string(name) + " (reused material)");
            
        return index;
    }

    mi::Uint32 Mdl_target::Resource_callback::get_string_index(
        mi::neuraylib::IValue_string const *s)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
            m_target->get_target_code());

        for (mi::Size i = 0, n = target_code->get_string_constant_count(); i < n; ++i)
            if (strcmp(target_code->get_string_constant(i), s->get_value()) == 0)
                return static_cast<mi::Uint32>(i);

        log_error("TODO: String constant not found.", SRC);
        return 0;
    }

    // --------------------------------------------------------------------------------------------

    Mdl_transaction::Mdl_transaction(Mdl_sdk* sdk)
        : m_sdk(sdk)
        , m_transaction_mtx()
    {
        mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
        m_transaction = scope->create_transaction();
    }

    Mdl_transaction::~Mdl_transaction()
    {
        std::lock_guard<std::mutex> lock(m_transaction_mtx);
        m_transaction->commit();
        m_transaction = nullptr;
    }

    void Mdl_transaction::commit()
    {
        std::lock_guard<std::mutex> lock(m_transaction_mtx);
        m_transaction->commit();
        mi::base::Handle<mi::neuraylib::IScope> scope(m_sdk->get_database().get_global_scope());
        m_transaction = scope->create_transaction();
    }

    // --------------------------------------------------------------------------------------------

    static std::atomic<size_t> sa_target_code_id = 0;
    Mdl_target::Mdl_target(Base_application* app, Mdl_sdk* sdk)
        : m_app(app)
        , m_sdk(sdk)
        , m_id(sa_target_code_id.fetch_add(1))
        , m_target_code(nullptr)
        , m_hlsl_source_code("")
        , m_dxil_compiled_library(nullptr)
        , m_read_only_data_segment(nullptr)
        , m_min_texture_count(0)
        , m_resource_names()
        , m_finalized(false)
    {
        // add the empty texture
        m_resource_names.emplace_back("");

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());
        m_link_unit = m_sdk->get_transaction().execute<mi::neuraylib::ILink_unit*>(
            [&](mi::neuraylib::ITransaction* t) {
                return m_sdk->get_backend().create_link_unit(t, context.get());
            });

        if (!m_sdk->log_messages(context.get()))
        {
            log_error("MDL creating a link unit failed.", SRC);
        }
    }

    Mdl_target::~Mdl_target()
    {
        m_target_code = nullptr;
        m_link_unit = nullptr;
        m_dxil_compiled_library = nullptr;

        if (m_read_only_data_segment) delete m_read_only_data_segment;
    }

    const mi::neuraylib::ITarget_code* Mdl_target::get_target_code() const
    {
        if (!m_target_code)
            return nullptr;
            
        m_target_code->retain();
        return m_target_code.get();
    }

    void Mdl_target::add_on_generated(std::function<bool(D3DCommandList*)> callback)
    {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        m_on_generated_callbacks.push_back(callback);
    }

    size_t Mdl_target::update_min_texture_count(size_t count)
    {
        if (m_min_texture_count < count)
            m_min_texture_count = count;
        return m_min_texture_count;
    }

    bool Mdl_target::generate()
    {
        bool expected = false;
        if (!m_finalized.compare_exchange_strong(expected, true)) 
        {
            log_error("Target code was already generated.", SRC);
            return false;
        }

        // since this method can be called from multiple threads simultaneously
        // a new context for is created
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(m_sdk->create_context());

        {
            Timing t("generating target code (id: " + std::to_string(m_id) + ")");
            m_target_code = m_sdk->get_backend().translate_link_unit(
                m_link_unit.get(), context.get());
        }

        std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        if (!m_sdk->log_messages(context.get()))
        {
            log_error("MDL target code generation failed.", SRC);
            return false;
        }

        Timing t2("loading MDL resources (id: " + std::to_string(m_id) + ")");

        // create a command list for uploading data to the GPU
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        // add all textures known to the link unit
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

        // ... in parallel, if not forced otherwise
        std::vector<std::thread> tasks;
        std::atomic_bool success = true;
        for (auto&& cb : m_on_generated_callbacks)
        {
            // sequentially
            if (m_app->get_options()->force_single_theading)
            {
                if (!cb(command_list))
                    success.store(false);
            }
            // asynchronously
            else
            {
                tasks.emplace_back(std::thread([&, cb]() 
                {
                    // no not fill command lists from different threads
                    D3DCommandList* local_command_list = command_queue->get_command_list();

                    if (!cb(local_command_list))
                        success.store(false);

                    command_queue->execute_command_list(local_command_list);
                }));

            }
        }

        // wait for all loading tasks
        for (auto &t : tasks)
            t.join();

        // any errors?
        if (!success.load())
        {
            log_error("On generate code callback return with failure.", SRC);
            return false;
        }

        // in order to load resources in parallel a continuous block of resource handles
        // for this target_code is allocated
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
        size_t handle_count = 1;                        // read-only segment
        handle_count += m_resource_names.size() - 1;    // valid textures
        // ...                                          // light profiles, ...

        Descriptor_heap_handle first_handle = resource_heap.reserve_views(handle_count);
        if (!first_handle.is_valid()) 
            return false;

        // keep the first handle, this is the offset into the applications global descriptor heap
        m_first_descriptor_heap_gpu_handle = first_handle.get_gpu_handle();

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

        // create resource view on the heap (at the first position of the target codes block)
        if (!resource_heap.create_shader_resource_view(
            m_read_only_data_segment, true, first_handle))
            return false;

        // copy data to the GPU
        if (m_read_only_data_segment && !m_read_only_data_segment->upload(command_list))
            return false;

        size_t n = m_resource_names.size();
        std::vector<Texture*> textures;
        textures.resize(n - 1, nullptr);

        // load the texture in parallel, if not forced otherwise
        // skip the invalid texture that is always present
        tasks.clear();
        for (size_t t = 1; t < n; ++t)
        {
            const char* texture_name = m_resource_names[t].c_str();
            log_info(
                "target code id: " + std::to_string(get_id()) +
                " - texture id: " + std::to_string(t) +
                " - resource: " + std::string(texture_name));

            // load sequentially
            if (m_app->get_options()->force_single_theading)
            {
                textures[t-1] = m_sdk->get_library()->access_texture_resource(
                    texture_name, command_list);
            }
            // load asynchronously
            else
            {
                tasks.emplace_back(std::thread([&, t, texture_name]() 
                {
                    // no not fill command lists from different threads
                    D3DCommandList* local_command_list = command_queue->get_command_list();

                    textures[t-1] = m_sdk->get_library()->access_texture_resource(
                        texture_name, local_command_list);

                    command_queue->execute_command_list(local_command_list);
                }));
            }
        }

        // wait for all loading tasks
        for (auto &t : tasks)
            t.join();

        // create a resource view on the heap
        // starting at the second position in the block 
        for (size_t t = 1; t < n; ++t)
        { 
            // texture is null, loading failed
            if (!textures[t - 1])
                return false;

            Descriptor_heap_handle heap_handle = first_handle.create_offset(t); // offset t+1
            if (!heap_handle.is_valid())
                return false;

            if (!resource_heap.create_shader_resource_view(textures[t-1], heap_handle))
                return false;
        }

        // note that the offset in the heap starts with zero
        // for each target we set 'target_heap_region_start' in the local root signature

        // bind read-only data segment to shader
        m_resource_descriptor_table.register_srv(0, 2, 0);

        // bind textures to shader
        if(textures.size() > 0)
            m_resource_descriptor_table.register_srv(1, 2, 1, textures.size());

        // generate some dxr specific shader code to hook things up
        // -------------------------------------------------------------------

        // generate the actual shader code with the help of some snippets
        m_hlsl_source_code.clear();
        m_hlsl_source_code += "#define MDL_TARGET_REGISTER_SPACE space2\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_REGISTER_SPACE space3\n";

        m_hlsl_source_code += "#define MDL_RO_DATA_SEGMENT_SLOT t0\n";
        m_hlsl_source_code += "#define MDL_TARGET_TEXTURE_SLOT_BEGIN t1\n";
        m_hlsl_source_code += "#define MDL_TARGET_TEXTURE_SLOT_COUNT " + std::to_string(textures.size()) + "\n";

        m_hlsl_source_code += "#define MDL_ARGUMENT_BLOCK_SLOT t1\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_TEXTURE_SLOT_BEGIN t2\n";
        m_hlsl_source_code += "#define MDL_MATERIAL_TEXTURE_SLOT_COUNT " + std::to_string(m_min_texture_count) + "\n";

        m_hlsl_source_code += "#define MDL_TEXTURE_SAMPLER_SLOT s0\n";
        m_hlsl_source_code += "#define MDL_LATLONGMAP_SAMPLER_SLOT s1\n";
        m_hlsl_source_code += "#define MDL_NUM_TEXTURE_RESULTS " + 
                              std::to_string(m_sdk->get_num_texture_results()) + "\n";

        m_hlsl_source_code += "\n";
        if (m_app->get_options()->automatic_derivatives) m_hlsl_source_code += "#define USE_DERIVS\n";
        if (m_app->get_options()->enable_auxiliary) m_hlsl_source_code += "#define ENABLE_AUXILIARY\n";

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

        std::string auxiliary_switch_function[2] = {
            "void mdl_bsdf_auxiliary(in uint function_index, inout Bsdf_auxiliary_data sret_ptr, "
            "in Shading_state_material state) {\n"
            "   switch(function_index) {\n",

            "void mdl_edf_auxiliary(in uint function_index, inout Edf_auxiliary_data sret_ptr, "
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

                    case mi::neuraylib::ITarget_code::FK_DF_AUXILIARY:
                        auxiliary_switch_function[index] +=
                            "       case " + std::to_string(f-4) + ": " + 
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

            auxiliary_switch_function[i] +=
                "       default: break;\n"
                "   }\n"
                "}\n\n";
            m_hlsl_source_code += auxiliary_switch_function[i];
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
        file_stream.open(get_executable_folder() + "/link_unit_code_" + std::to_string(m_id) + ".hlsl");
        if (file_stream)
        {
            file_stream << m_hlsl_source_code.c_str();
            file_stream.close();
        }

        command_queue->execute_command_list(command_list);

        m_target_code = nullptr;
        m_link_unit = nullptr;
        return true;
    }

    bool Mdl_target::compile() 
    {
        // generate has be called first
        if (m_link_unit != nullptr)
        {
            log_error("Compiling HLSL target code not possible before generation. Target ID: "
                + std::to_string(m_id), SRC);
            return false;
        }

        // compile to DXIL
        {
            Timing t("compiling HLSL to DXIL (id: " + std::to_string(m_id) + ")");

            std::map<std::string, std::string> defines;
            defines["TARGET_CODE_ID"] = std::to_string(m_id);

            Shader_compiler compiler;
            m_dxil_compiled_library = compiler.compile_shader_library_from_string(
                get_hlsl_source_code(), "link_unit_code_" + std::to_string(m_id),
                &defines);
        }

        return m_dxil_compiled_library != nullptr;
    }

    // --------------------------------------------------------------------------------------------
    
    namespace
    {
        static std::atomic<uint32_t> s_material_shared_id_counter = 0;
    }

    Mdl_material_shared::Mdl_material_shared()
        : m_material_shared_id(s_material_shared_id_counter.fetch_add(1))
        , m_argument_layout_index(static_cast<mi::Size>(-1))
        , m_target(nullptr)
        , m_max_texture_count(0)
    {
    }
    
    // --------------------------------------------------------------------------------------------

    namespace {
        static std::atomic<uint32_t> s_material_id_counter = 0;
    }

    Mdl_material::Mdl_material(Base_application* app, const IScene_loader::Material& material_desc)
        : m_app(app)
        , m_shared(nullptr)
        , m_name(material_desc.name)
        , m_material_id(s_material_id_counter.fetch_add(1))
        , m_flags(IMaterial::Flags::None)
        , m_constants(m_app, m_name + "_Constants")
        , m_argument_block_data(nullptr) // size is not known at this point
        , m_info(nullptr)
        , m_resource_names()
    {
        // add the empty texture
        m_resource_names.emplace_back("");
        m_constants.data.material_id = m_material_id;
    }


    Mdl_material::~Mdl_material()
    {
        if (m_argument_block_data) delete m_argument_block_data;
        if (m_info) delete m_info;
        m_shared = nullptr;
    }

    namespace
    {
        static Descriptor_table s_resource_descriptor_table;
        static std::atomic_bool s_resource_descriptor_table_filled = false;
    }

    const Descriptor_table& Mdl_material::get_static_descriptor_table()
    {
        // note that the offset in the heap starts with zero
        // for each material we set 'material_heap_region_start' in the local root signature
        bool expected = false;
        if (s_resource_descriptor_table_filled.compare_exchange_strong(expected, true))
        {
            // bind material constants
            s_resource_descriptor_table.register_cbv(0, 3, 0);

            // bind material argument block 
            s_resource_descriptor_table.register_srv(1, 3, 1);
        }

        return s_resource_descriptor_table;
    }

    const Descriptor_table Mdl_material::get_descriptor_table()
    {
        // same as the static case + additional per material resources
        Descriptor_table table(get_static_descriptor_table());

        // bind per material resources
        if(m_shared->m_max_texture_count > 0)
            table.register_srv(2, 3, 2, m_shared->m_max_texture_count);

        return std::move(table);
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
        const mi::neuraylib::IMaterial_definition* material_definition,
        const mi::neuraylib::ICompiled_material* compiled_material,
        bool first_instance,
        D3DCommandList* command_list)
    {
        auto after_clean_up = [&](bool result)
        {
            // drop mdl reference
            material_definition->release();
            compiled_material->release();
            return result;
        };

        // get target code infos for this specific material
        Mdl_target* target = m_shared->m_target;
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(target->get_target_code());
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
                    &target->get_resource_callback(m_app->get_options()->share_target_code 
                        ? nullptr  // when sharing a target code, the resources are added there
                        : this));  // when not, the material itself stores its resources 

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

        // reserve/create resource views
        Descriptor_heap& resource_heap = *m_app->get_resource_descriptor_heap();
        m_first_descriptor = resource_heap.reserve_views(
            1 + // for constant buffer
            1 + // for argument block
            m_resource_names.size() - 1); // for per material textures (>= 1)
        if (!m_first_descriptor.is_valid())
            return after_clean_up(false);

        // keep the first handle, 
        // this is the materials offset into the applications global descriptor heap
        m_descriptor_heap_region = m_first_descriptor.get_gpu_handle();

        // create the actual constant buffer view
        if (!resource_heap.create_constant_buffer_view(&m_constants, m_first_descriptor))
            return after_clean_up(false);

        // create a resource view for the argument block
        auto argument_block_data_srv = m_first_descriptor.create_offset(1);
        if (!argument_block_data_srv.is_valid()) 
            return after_clean_up(false);

        if (!resource_heap.create_shader_resource_view(
            m_argument_block_data, true, argument_block_data_srv))
                return after_clean_up(false);

        // load per material textures
        size_t n = m_resource_names.size();
        std::vector<Texture*> textures; 
        textures.resize(n - 1, nullptr);

        // load the texture in parallel, if not forced otherwise
        // skip the invalid texture that is always present
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        std::vector<std::thread> tasks;
        for (size_t t = 1; t < n; ++t)
        {
            const char* texture_name = m_resource_names[t].c_str();

            // load sequentially
            if (m_app->get_options()->force_single_theading)
            {
                textures[t - 1] = m_app->get_mdl_sdk().get_library()->access_texture_resource(
                    texture_name, command_list);
            }
            // load asynchronously
            else
            {
                tasks.emplace_back(std::thread([&, t, texture_name]() 
                {
                    // no not fill command lists from different threads
                    D3DCommandList* local_command_list = command_queue->get_command_list();

                    textures[t - 1] = m_app->get_mdl_sdk().get_library()->access_texture_resource(
                        texture_name, local_command_list);

                    command_queue->execute_command_list(local_command_list);
                }));
            }
        }

        // wait for all loading tasks
        for (auto &t : tasks)
            t.join();

        // create a resource view on the heap
        // starting at the second position in the block 
        for (size_t t = 1; t < n; ++t)
        {
            // texture is null, loading failed
            if(!textures[t - 1])
                return after_clean_up(false);

            Descriptor_heap_handle heap_handle = m_first_descriptor.create_offset(t+1); // offset t+2
            if (!heap_handle.is_valid())
                return after_clean_up(false);

            if (!resource_heap.create_shader_resource_view(textures[t - 1], heap_handle))
                return after_clean_up(false);
        }

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
