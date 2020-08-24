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

#include "example_shared.h"
#include "utils.h"

#include <mi/mdl_sdk.h>
#include <mi/neuraylib/definition_wrapper.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

namespace
{
    std::atomic<uint64_t> s_material_desc_id_counter(0);
} // anonymous

// ------------------------------------------------------------------------------------------------

const std::string Mdl_material_description::Invalid_material_identifier = "::dxr::not_available";

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description(const std::string& unique_material_identifier)
    : m_parameters()
    , m_is_loaded(false)
    , m_is_fallback(false)
    , m_supports_reloading(false)
    , m_loader(nullptr)
    , m_parameter_list(nullptr)
    , m_module_db_name("")
    , m_material_defintion_db_name("")
    , m_qualified_module_name("")
    , m_material_name("")
    , m_name_in_scene("")
    , m_source_code("")
    , m_flags(IMaterial::Flags::None)
    , m_unique_id(s_material_desc_id_counter.fetch_add(1))
{
    m_parameters.name = unique_material_identifier;
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description()
    : Mdl_material_description(Invalid_material_identifier)
{
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::Mdl_material_description(const IScene_loader::Material& description)
    : Mdl_material_description()
{
    m_parameters = description;
}

// ------------------------------------------------------------------------------------------------

Mdl_material_description::~Mdl_material_description()
{
    m_parameter_list = nullptr;
}

// ------------------------------------------------------------------------------------------------

const IScene_loader::Material& Mdl_material_description::get_material_parameters() const
{
    return m_parameters;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // loaded before
    if (m_is_loaded)
        return true;

    // special case: the invalid material is requested directly
    if (m_parameters.name == Invalid_material_identifier)
    {
        if (!load_material_definition_fallback(sdk, scene_directory , context))
        {
            log_error("[FATAL] Invalid (fall-back) material can not be loaded.", SRC);
            return false;
        }

        m_is_loaded = true;
    }

    // depending on the type of material override (simple name convention)
    // it can make sense to use a gltf material in case of loading failure
    bool skip_glft_material_as_first_fall_back = false;

    // handle mdl (by convention when the name starts with '::')
    if (!m_is_loaded && mi::examples::strings::starts_with(m_parameters.name, "::"))
    {
        log_info("Material name matches the convention for overriding by an MDL material: " +
            m_parameters.name);

        m_is_loaded = load_material_definition_mdl(sdk, scene_directory, context);
        skip_glft_material_as_first_fall_back = true;
    }

    // handle mdle
    if (!m_is_loaded && (
        mi::examples::strings::ends_with(m_parameters.name, ".mdle") ||
        mi::examples::strings::ends_with(m_parameters.name, ".mdle::main")))
    {
        log_info("Material name matches the convention for overriding by an MDLE material: " +
            m_parameters.name);

        m_is_loaded = load_material_definition_mdle(sdk, scene_directory, context);
        skip_glft_material_as_first_fall_back = true;
    }

    // handle loaders
    if (!m_is_loaded)
    {
        sdk.get_library()->visit_material_description_loaders(
            [&](const IMdl_material_description_loader* loader)
            {
                // load if there is a match
                if (loader->match_gltf_name(m_parameters.name))
                {
                    log_info("Material name matches the convention for overriding: " +
                        m_parameters.name);

                    m_loader = loader; // keep the loader for reloading
                    m_is_loaded = load_material_definition_loader(
                        sdk, scene_directory, context, loader);

                    // do not use the gltf support as fall-back (pink instead)
                    skip_glft_material_as_first_fall_back = true;
                    return false; // stop visits, even in case of failure
                }
                return true; // continue visits
            });
    }

    // fall-back to gltf support materials
    if (!skip_glft_material_as_first_fall_back && !m_is_loaded)
    {
        m_is_loaded = load_material_definition_gltf_support(sdk, scene_directory, context);
        log_info("Material name does not match any convention, "
            "so the GLTF support material is used: " + m_parameters.name);
    }

    // fall-back
    if (!m_is_loaded)
        m_is_loaded = load_material_definition_fallback(sdk, scene_directory, context);

    if (m_is_loaded)
    {
        m_material_defintion_db_name = m_module_db_name + "::" + m_material_name;

        // reflect changed imports
        sdk.get_library()->update_module_dependencies(m_module_db_name);

        return true;
    }

    log_error("[FATAL] Material definition can't be loaded and fall-back solutions failed for: "
        + m_parameters.name, SRC);
    return false;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::is_loaded() const
{
    return m_is_loaded;
}

// ------------------------------------------------------------------------------------------------

const mi::neuraylib::IExpression_list* Mdl_material_description::get_parameters() const
{
    if (!m_parameter_list)
        return nullptr;

    m_parameter_list->retain();
    return m_parameter_list.get();
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_qualified_module_name() const
{
    return m_is_loaded ? m_qualified_module_name.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_material_name() const
{
    return m_is_loaded ? m_material_name.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_scene_name() const
{
    return m_is_loaded ? m_name_in_scene.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

IMaterial::Flags Mdl_material_description::get_flags() const
{
    return m_is_loaded ? m_flags : IMaterial::Flags::None;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::is_fallback() const
{
    return m_is_loaded ? m_is_fallback : false;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::supports_reloading() const
{
    return m_is_loaded ? m_supports_reloading : false;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_module_db_name() const
{
    return m_is_loaded ? m_module_db_name.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::get_material_defintion_db_name() const
{
    return m_is_loaded ? m_material_defintion_db_name.c_str() : nullptr;
}

// ------------------------------------------------------------------------------------------------

const char* Mdl_material_description::regenerate_source_code(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // not supported or no module that was loaded from source
    if (!m_is_loaded || !m_supports_reloading || m_source_code.empty())
        return nullptr;

    // materials with generated code
    std::string generated_code = "";

    // handle loader
    if (m_loader)
        generated_code = m_loader->generate_mdl_source_code(m_parameters.name, scene_directory);

    // code was generated successfully
    if (!generated_code.empty())
    {
        m_source_code = generated_code;
        return m_source_code.c_str();
    }

    // code generation failed, or no generator for this type of material
    load_material_definition_fallback(sdk, scene_directory, context);
    return nullptr;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_mdle(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // drop the optional module name
    if (mi::examples::strings::ends_with(m_parameters.name, ".mdle::main"))
        m_parameters.name = m_parameters.name.substr(0, m_parameters.name.length() - 6);

    // resolve relative paths within the scene directory
    if (mi::examples::io::is_absolute_path(m_parameters.name))
        m_qualified_module_name = m_parameters.name;
    else
        m_qualified_module_name = scene_directory + "/" + m_parameters.name;

    // check if the file exists
    if (!mi::examples::io::file_exists(m_qualified_module_name))
    {
        log_warning(
            "The referenced MDLE file does not exist: " + m_parameters.name +
            "\nUsing a fall-back material instead.");
        return false;
    }

    m_material_name = "main";

    // name only for display
    std::string name = m_qualified_module_name.substr(
        m_qualified_module_name.find_last_of('/') + 1);
    name = name.substr(0, name.length() - 4);
    m_name_in_scene = "[mdle] " + name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // check if the mdle contains a material
    std::string material_db_name = m_module_db_name + "::" + m_material_name;
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_defintion(
        sdk.get_transaction().access<const mi::neuraylib::IMaterial_definition>(
            material_db_name.c_str()));
    if (!material_defintion)
    {
        log_warning(
            "The referenced MDLE file does not contain a material: " +
            m_qualified_module_name + "\nUsing a fall-back material instead.");
        return false;
    }

    // mdle have defaults for all parameters
    // TODO: they can be overridden, possibly by data from the loaded scene
    m_parameter_list = nullptr;
    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_mdl(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // split module and material name
    // [::<package>]::<module>::<material>
    if (!mi::examples::mdl::parse_cmd_argument_material_name(
        m_parameters.name, m_qualified_module_name, m_material_name, false))
    {
        log_warning(
            "Material name is not a fully qualified material name: " + m_parameters.name +
            "\nUsing a fall-back material instead.");
        return false;
    }

    // name only for display
    m_name_in_scene = "[mdl] " + m_material_name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // database name of the material
    std::string material_definition_db_name = m_module_db_name + "::" + m_material_name;

    // check if the module contains the requested material
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> definition(
        sdk.get_transaction().access<const mi::neuraylib::IMaterial_definition>(
            material_definition_db_name.c_str()));
    if (!definition)
    {
        m_module_db_name = "";
        return false;
    }

    // create a parameter list for all parameters (even without default)
    // for that the Definition_wrapper can also be used when creating an instance
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
    {
        mi::base::Handle<const  mi::neuraylib::IType_list> parameter_types(
            definition->get_parameter_types());
        mi::base::Handle<const  mi::neuraylib::IExpression_list> defaults(
            definition->get_defaults());

        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            sdk.get_factory().create_value_factory(t));
        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            sdk.get_factory().create_expression_factory(t));

        m_parameter_list = ef->create_expression_list();

        mi::Size count = definition->get_parameter_count();
        for (mi::Size i = 0; i < count; ++i) {
            const char* name = definition->get_parameter_name(i);
            mi::base::Handle<const mi::neuraylib::IExpression> default_(defaults->get_expression(name));
            if (!default_) {
                mi::base::Handle<const mi::neuraylib::IType> type(parameter_types->get_type(i));
                mi::base::Handle<mi::neuraylib::IValue> value(vf->create(type.get()));
                mi::base::Handle<mi::neuraylib::IExpression> expr(ef->create_constant(value.get()));
                m_parameter_list->add_expression(name, expr.get());
            }
        }
    });

    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_gltf_support(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    m_qualified_module_name = "::nvidia::sdk_examples::gltf_support";

    // .. but it will be disabled for opaque material instances
    if (m_parameters.alpha_mode == IScene_loader::Material::Alpha_mode::Opaque)
        m_flags = mi::examples::enums::set_flag(m_flags, IMaterial::Flags::Opaque);

    if (m_parameters.single_sided == true)
        m_flags = mi::examples::enums::set_flag(m_flags, IMaterial::Flags::SingleSided);

    // model dependent support material name
    switch (m_parameters.pbr_model)
    {
        case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
            m_material_name = "gltf_material_khr_specular_glossiness";
            break;

        case IScene_loader::Material::Pbr_model::Metallic_roughness:
        default:
            m_material_name = "gltf_material";
            break;
    }

    // name only for display
    m_name_in_scene = "[gltf] " + m_parameters.name + " (" + std::to_string(m_unique_id) + ")";

    // load the module
    if (!load_mdl_module(sdk, scene_directory, context))
        return false;

    // setup the parameter list based on the gltf scene data
    parameterize_gltf_support_material(sdk, scene_directory, context);

    m_supports_reloading = true;
    m_is_fallback = false;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_mdl_module(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));

    // check if the module and thereby the material definition is already loaded
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

    // if not, load it
    if (!module)
    {
        // Load the module that contains the material.
        // This functions supports multi-threading. It blocks when the requested module
        // or a dependency is loaded by a different thread.
        sdk.get_impexp_api().load_module(
            sdk.get_transaction().get(), m_qualified_module_name.c_str(), context);

        // loading failed
        if (!sdk.log_messages(
            "Loading the module failed: " + m_qualified_module_name, context))
                return false;
    }

    m_module_db_name = module_db_name->get_c_str();
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_fallback(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // default module
    m_qualified_module_name = "::dxr";
    m_material_name = "not_available";

    static const char* default_module_src =
        "mdl 1.2;\n"
        "import df::*;\n"
        "export material not_available() = material(\n"
        "    surface: material_surface(\n"
        "        df::diffuse_reflection_bsdf(\n"
        "            tint: color(0.8, 0.0, 0.8)\n"
        "        )\n"
        "    )"
        ");";

    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));

    return sdk.get_transaction().execute<bool>([&](auto t)
    {
        // check if the module and thereby the material definition is already loaded
        mi::base::Handle<const mi::neuraylib::IModule> module(
            t->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

        // module is not loaded already
        if (!module)
        {
            // load it
            sdk.get_impexp_api().load_module_from_string(
                t, m_qualified_module_name.c_str(), default_module_src, context);

            // loading failed?
            if (!sdk.log_messages(
                "Loading the module failed: " + m_qualified_module_name, context, SRC))
                    return false;
        }

        // module is loaded now
        m_module_db_name = module_db_name->get_c_str();
        m_name_in_scene = "[mdl] invalid material (" + std::to_string(m_unique_id) + ")";

        m_supports_reloading = false;
        m_is_fallback = true;
        m_parameter_list = nullptr;
        m_flags = IMaterial::Flags::None;
        return true;
    });
}

// ------------------------------------------------------------------------------------------------

void Mdl_material_description::parameterize_gltf_support_material(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context)
{
    // this creates elements in the database, therefore it has to be locked when using threads
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
    {
        // create material parameters
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            sdk.get_neuray().get_api_component<mi::neuraylib::IMdl_factory>());
        mi::base::Handle<mi::neuraylib::IValue_factory> vf(
            mdl_factory->create_value_factory(t));
        mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
            mdl_factory->create_expression_factory(t));
        mi::base::Handle<mi::neuraylib::IType_factory> tf(
            mdl_factory->create_type_factory(t));

        // create a new parameter list
        m_parameter_list = ef->create_expression_list();

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
            std::string file_path = scene_directory + "/" + releative_texture_path;

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

            m_parameter_list->add_expression(expression_name.c_str(), expr.get());

        };

        // helper to add a color parameter
        auto add_color = [&](
            const std::string& expression_name,
            float r, float g, float b)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_color(r, g, b));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            m_parameter_list->add_expression(expression_name.c_str(), expr.get());
        };

        // helper to add a float
        auto add_float = [&](
            const std::string& expression_name,
            float x)
        {
            mi::base::Handle<mi::neuraylib::IValue> value(vf->create_float(x));
            mi::base::Handle<mi::neuraylib::IExpression> expr(
                ef->create_constant(value.get()));

            m_parameter_list->add_expression(expression_name.c_str(), expr.get());
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

            m_parameter_list->add_expression(expression_name.c_str(), expr.get());
        };

        // helper to add clear-coat
        auto add_clearcoat = [&](
            const mdl_d3d12::IScene_loader::Material::Model_data_materials_clearcoat& clearcoat)
        {
            add_float("clearcoat_factor", clearcoat.clearcoat_factor);
            add_texture("clearcoat_texture", clearcoat.clearcoat_texture, 1.0f);
            add_float("clearcoat_roughness_factor", clearcoat.clearcoat_roughness_factor);
            add_texture("clearcoat_roughness_texture", clearcoat.clearcoat_roughness_texture, 1.0f);
            add_texture("clearcoat_normal_texture", clearcoat.clearcoat_normal_texture, 1.0f);
        };

        // add the actual parameters to the parameter list
        add_texture("normal_texture", m_parameters.normal_texture, 1.0f);
        add_float("normal_scale_factor", m_parameters.normal_scale_factor);

        add_texture("occlusion_texture", m_parameters.occlusion_texture, 1.0f);
        add_float("occlusion_strength", m_parameters.occlusion_strength);

        add_texture("emissive_texture", m_parameters.emissive_texture, 2.2f);
        add_color("emissive_factor", m_parameters.emissive_factor.x,
                    m_parameters.emissive_factor.y, m_parameters.emissive_factor.z);

        add_enum("alpha_mode", static_cast<mi::Sint32>(m_parameters.alpha_mode));
        add_float("alpha_cutoff", m_parameters.alpha_cutoff);

        // model dependent parameters
        switch (m_parameters.pbr_model)
            {
            case IScene_loader::Material::Pbr_model::Khr_specular_glossiness:
            {
                add_texture("diffuse_texture",
                            m_parameters.khr_specular_glossiness.diffuse_texture, 2.2f);

                add_color("diffuse_factor",
                            m_parameters.khr_specular_glossiness.diffuse_factor.x,
                            m_parameters.khr_specular_glossiness.diffuse_factor.y,
                            m_parameters.khr_specular_glossiness.diffuse_factor.z);

                add_float("base_alpha",
                            m_parameters.khr_specular_glossiness.diffuse_factor.w);

                add_texture("specular_glossiness_texture",
                            m_parameters.khr_specular_glossiness.specular_glossiness_texture, 2.2f);

                add_color("specular_factor",
                            m_parameters.khr_specular_glossiness.specular_factor.x,
                            m_parameters.khr_specular_glossiness.specular_factor.y,
                            m_parameters.khr_specular_glossiness.specular_factor.z);
                add_float("glossiness_factor",
                            m_parameters.khr_specular_glossiness.glossiness_factor);
                return;
            }

            case IScene_loader::Material::Pbr_model::Metallic_roughness:
            default:
            {
                add_texture("base_color_texture",
                            m_parameters.metallic_roughness.base_color_texture, 2.2f);

                add_color("base_color_factor",
                            m_parameters.metallic_roughness.base_color_factor.x,
                            m_parameters.metallic_roughness.base_color_factor.y,
                            m_parameters.metallic_roughness.base_color_factor.z);

                add_float("base_alpha",
                            m_parameters.metallic_roughness.base_color_factor.w);

                add_texture("metallic_roughness_texture",
                            m_parameters.metallic_roughness.metallic_roughness_texture, 1.0f);

                add_float("metallic_factor",
                            m_parameters.metallic_roughness.metallic_factor);

                add_float("roughness_factor",
                            m_parameters.metallic_roughness.roughness_factor);

                add_clearcoat(m_parameters.metallic_roughness.clearcoat);
                return;
            }
        }
    });
}

// -------------------------------------------------------------------------------------------------

bool Mdl_material_description::load_material_definition_loader(
    Mdl_sdk& sdk,
    const std::string& scene_directory,
    mi::neuraylib::IMdl_execution_context* context,
    const IMdl_material_description_loader* loader)
{
    // naming convention using the gltf material name attribute
    std::string gltf_name = m_parameters.name;

    // generate the source code
    m_source_code = loader->generate_mdl_source_code(gltf_name, scene_directory);
    if (m_source_code.empty())
        return false;

    // compute a full qualified module name
    std::string module_name = gltf_name;
    std::replace(module_name.begin(), module_name.end(), '\\', '_');
    std::replace(module_name.begin(), module_name.end(), '/', '_');
    std::replace(module_name.begin(), module_name.end(), ':', '_');
    std::replace(module_name.begin(), module_name.end(), '.', '_');
    if (module_name.length() < 2 || module_name[0] != ':' || module_name[1] == ':')
        module_name = "::" + module_name;
    m_qualified_module_name = module_name;

    // load the actual module, sequentially for now
    sdk.get_transaction().execute<void>([&](mi::neuraylib::ITransaction* t)
        {
            auto& mdl_impexp = sdk.get_impexp_api();
            mdl_impexp.load_module_from_string(
                t, m_qualified_module_name.c_str(),
                m_source_code.c_str(),
                context);
        });
    if (!sdk.log_messages(
        "Loading generated material (from materialX) failed: " + m_parameters.name,
        context, SRC))
            return false;

    // expected database name of the module to load
    mi::base::Handle<const mi::IString> module_db_name(
        sdk.get_factory().get_db_module_name(m_qualified_module_name.c_str()));
    m_module_db_name = module_db_name->get_c_str();

    // get the loaded module
    mi::base::Handle<const mi::neuraylib::IModule> module(
        sdk.get_transaction().access<const mi::neuraylib::IModule>(
            module_db_name->get_c_str()));

    if (module->get_material_count() == 0)
    {
        log_error("Generated MDL exports no material: " + m_parameters.name, SRC);
        return false;
    }

    // get the first material in the module
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        sdk.get_transaction().access<const mi::neuraylib::IMaterial_definition>(module->get_material(0)));
    m_material_name = material_definition->get_mdl_simple_name();

    // set the name in the scene graph and on the GUI
    m_name_in_scene = loader->get_scene_name_prefix() + " " + m_material_name +
        " (" + std::to_string(m_unique_id) + ")";

    m_supports_reloading = loader->supports_reload();
    m_parameter_list = nullptr;         // defaults will be used
    m_flags = IMaterial::Flags::None;   // no hard coded assumptions
    return true;
}

}}} // mi::examples::mdl_d3d12
