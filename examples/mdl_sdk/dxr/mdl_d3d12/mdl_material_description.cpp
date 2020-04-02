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

#include "mdl_material_description.h"
#include "mdl_material.h"
#include "mdl_sdk.h"
#include "base_application.h"
#include <mi/mdl_sdk.h>

#include "example_shared.h"

namespace mdl_d3d12
{
    Mdl_material_description::Mdl_material_description(
        Base_application* app,
        IScene_loader::Material description)
        : m_app(app)
        , m_description(description)
        , m_qualified_module_name("")
        , m_material_name("")
        , m_flags(IMaterial::Flags::None)
    {

        // parse the mdl name
        bool is_mdle = m_description.name.rfind(".mdle") != std::string::npos;
        if (is_mdle)
        {
            // resolve relative paths within the scene directory
            if (is_absolute_path(m_description.name))
                m_qualified_module_name = m_description.name;
            else
                m_qualified_module_name =
                app->get_options()->scene_directory + "/" + m_description.name;

            m_material_name = "main";
        }
        else
        {
            // regular MDL name (fully qualified (absolute) mdl material name)
            // [::<package>]::<module>::<material>
            size_t sep_pos = m_description.name.rfind("::");
            if (str_starts_with(m_description.name, "::") && sep_pos != 0)
            {
                m_qualified_module_name = m_description.name.substr(0, sep_pos);
                m_material_name = m_description.name.substr(sep_pos + 2);
            }
            // handle none mdl materials
            else
            {
                // TODO
                // use a default materials that can deal with infos in the material description of
                // the scene file, e.g. GLTF material parameters

                m_qualified_module_name = "::nvidia::sdk_examples::gltf_support";

                // .. but it will be disabled for opaque material instances
                if (m_description.alpha_mode == IScene_loader::Material::Alpha_mode::Opaque)
                    m_flags = m_flags | IMaterial::Flags::Opaque;

                if (m_description.single_sided == true)
                    m_flags = m_flags | IMaterial::Flags::SingleSided;

                // model dependent support material name
                switch (m_description.pbr_model)
                {
                    case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
                        m_material_name = "gltf_material_khr_specular_glossiness";
                        break;

                    case IScene_loader::Material::Pbr_model::Metallic_roughness:
                    default:
                        m_material_name = "gltf_material";
                        break;
                }
            }
        }

        // TODO check if that all worked out
        // if not, use a default pink error material
    }

    // --------------------------------------------------------------------------------------------

    const mi::neuraylib::IExpression_list* Mdl_material_description::get_parameters() const
    {
        // create mdl parameters for the GLTF support materials from GLTF parameters
        if (m_qualified_module_name == "::nvidia::sdk_examples::gltf_support")
            return parameterize_support_material();

        return nullptr;
    }

    // --------------------------------------------------------------------------------------------

    const mi::neuraylib::IExpression_list* 
        Mdl_material_description::parameterize_support_material() const
    {
        Mdl_sdk& mdl_sdk = m_app->get_mdl_sdk();
        mi::base::Handle<mi::neuraylib::IExpression_list> parameters(nullptr);

        // this creates elements in the database, therefore it has to be locked when using threads
        mdl_sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
        {
            // create material parameters
            mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
                mdl_sdk.get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());
            mi::base::Handle<mi::neuraylib::IValue_factory> vf(
                mdl_factory->create_value_factory(t));
            mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
                mdl_factory->create_expression_factory(t));
            mi::base::Handle<mi::neuraylib::IType_factory> tf(
                mdl_factory->create_type_factory(t));

            // create a new parameter list
            parameters = mi::base::Handle<mi::neuraylib::IExpression_list>(
                ef->create_expression_list());

            // helper to add a texture if it is available
            auto add_texture = [&](
                const std::string& expression_name,
                const std::string& releative_texture_path, float gamma)
            {
                if (releative_texture_path.empty()) return;

                // TODO handle textures that already in the DB because they have been added before

                mi::base::Handle<mi::neuraylib::IImage> image(
                    t->create<mi::neuraylib::IImage>("Image"));
                std::string image_name = "mdl::" + releative_texture_path + "_image";

                std::string file_path = 
                    m_app->get_options()->scene_directory + "/" + releative_texture_path;

                image->reset_file(file_path.c_str());
                t->store(image.get(), image_name.c_str());

                mi::base::Handle<mi::neuraylib::ITexture> texture(
                    t->create<mi::neuraylib::ITexture>("Texture"));
                texture->set_image(image_name.c_str());
                texture->set_gamma(gamma);
                std::string texture_name = "mdl::" + releative_texture_path + "_texture2d";
                t->store(texture.get(), texture_name.c_str());

                // Mark the texture for removing right away.
                // Note, this will not delete the data immediately. Instead it will be deleted
                // with the next transaction::commit(). Until then, we will have copied to resources
                // to the GPU.
                t->remove(texture_name.c_str());
                t->remove(image_name.c_str());

                mi::base::Handle<const mi::neuraylib::IType_texture> type(
                    tf->create_texture(mi::neuraylib::IType_texture::TS_2D));
                mi::base::Handle<mi::neuraylib::IValue_texture> value(
                    vf->create_texture(type.get(), texture_name.c_str()));
                mi::base::Handle<mi::neuraylib::IExpression> expr(
                    ef->create_constant(value.get()));

                parameters->add_expression(expression_name.c_str(), expr.get());

            };

            // helper to add a color parameter
            auto add_color = [&](
                const std::string& expression_name,
                float r, float g, float b)
            {
                mi::base::Handle<mi::neuraylib::IValue> value(vf->create_color(r, g, b));
                mi::base::Handle<mi::neuraylib::IExpression> expr(
                    ef->create_constant(value.get()));

                parameters->add_expression(expression_name.c_str(), expr.get());
            };

            // helper to add a float
            auto add_float = [&](
                const std::string& expression_name,
                float x)
            {
                mi::base::Handle<mi::neuraylib::IValue> value(vf->create_float(x));
                mi::base::Handle<mi::neuraylib::IExpression> expr(
                    ef->create_constant(value.get()));

                parameters->add_expression(expression_name.c_str(), expr.get());
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

                parameters->add_expression(expression_name.c_str(), expr.get());
            };

            // add the actual parameters to the parameter list
            add_texture("normal_texture", m_description.normal_texture, 1.0f);
            add_float("normal_scale_factor", m_description.normal_scale_factor);

            add_texture("occlusion_texture", m_description.occlusion_texture, 1.0f);
            add_float("occlusion_strength", m_description.occlusion_strength);

            add_texture("emissive_texture", m_description.emissive_texture, 2.2f);
            add_color("emissive_factor", m_description.emissive_factor.x,
                      m_description.emissive_factor.y, m_description.emissive_factor.z);

            add_enum("alpha_mode", static_cast<mi::Sint32>(m_description.alpha_mode));
            add_float("alpha_cutoff", m_description.alpha_cutoff);

            // model dependent parameters
            switch (m_description.pbr_model)
                {
                case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
                {
                    add_texture("diffuse_texture",
                                m_description.khr_specular_glossiness.diffuse_texture, 2.2f);

                    add_color("diffuse_factor",
                              m_description.khr_specular_glossiness.diffuse_factor.x,
                              m_description.khr_specular_glossiness.diffuse_factor.y,
                              m_description.khr_specular_glossiness.diffuse_factor.z);

                    add_float("base_alpha",
                              m_description.khr_specular_glossiness.diffuse_factor.w);

                    add_texture("specular_glossiness_texture",
                                m_description.khr_specular_glossiness.specular_glossiness_texture, 2.2f);

                    add_color("specular_factor",
                              m_description.khr_specular_glossiness.specular_factor.x,
                              m_description.khr_specular_glossiness.specular_factor.y,
                              m_description.khr_specular_glossiness.specular_factor.z);
                    add_float("glossiness_factor",
                              m_description.khr_specular_glossiness.glossiness_factor);
                    return;
                }

                case IScene_loader::Material::Pbr_model::Metallic_roughness:
                default:
                {
                    add_texture("base_color_texture",
                                m_description.metallic_roughness.base_color_texture, 2.2f);

                    add_color("base_color_factor",
                              m_description.metallic_roughness.base_color_factor.x,
                              m_description.metallic_roughness.base_color_factor.y,
                              m_description.metallic_roughness.base_color_factor.z);

                    add_float("base_alpha",
                              m_description.metallic_roughness.base_color_factor.w);

                    add_texture("metallic_roughness_texture",
                                m_description.metallic_roughness.metallic_roughness_texture, 1.0f);

                    add_float("metallic_factor",
                              m_description.metallic_roughness.metallic_factor);

                    add_float("roughness_factor",
                              m_description.metallic_roughness.roughness_factor);
                    return;
                }
            }
        });

        parameters->retain();
        return parameters.get();
    }
}

