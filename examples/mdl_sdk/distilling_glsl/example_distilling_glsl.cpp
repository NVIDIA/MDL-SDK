/******************************************************************************
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/distilling_glsl/example_distilling_glsl.cpp
//
// Shows how to distill an MDL material to UE4 and use the distilling result in a GLSL
// PBR shader by generating GLSL code for the relevant material expressions.
// When more than one material is specified on the command line, you can use the
// left, right arrows to switch between them.
// Using the up and down arrows you can switch between baked and non-baked
// GLSL expressions.

#include <map>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "example_shared.h"
#include "example_glsl_shared.h"
#include "example_distilling_shared.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define _USE_MATH_DEFINES
#include <math.h>


// This selects SSBO (Shader Storage Buffer Objects) mode for passing uniforms and MDL const data.
// Should not be disabled unless you only use materials with very small const data. The noise
// functions from the ::base module for example use lookup tables that account for a large amount
// of const data in the generated GLSL code.
#if !defined(MI_PLATFORM_MACOSX) && !defined(MI_ARCH_ARM_64)
    #define USE_SSBO
#endif

// If defined, the GLSL backend remap these functions
//   float ::base::perlin_noise(float4 pos)
//   float ::base::mi_noise(float3 pos)
//   float ::base::mi_noise(int3 pos)
//   ::base::worley_return ::base::worley_noise(float3 pos, float jitter, int metric)
//
// to lut-free alternatives. When enabled, you can avoid to set the USE_SSBO define for this
// example.
//
//#define REMAP_NOISE_FUNCTIONS

// Enable this to dump the generated GLSL code to stdout/file.
//#define DUMP_GLSL

// Application options
struct Options {

    bool show_window;
    bool bake;
    int baking_resolution_x;
    int baking_resolution_y;
    float exposure;
    std::vector<std::string> material_names;
    std::string hdrfile;
    std::string outputfile;

    Options()
        : show_window(true)
        , bake(true)
        , baking_resolution_x(2048)
        , baking_resolution_y(2048)
        , exposure(0.0f)
        , hdrfile("nvidia/sdk_examples/resources/environment.hdr")
        , outputfile("output.exr")
        {}
};

// Helper struct to ease passing commonly used objects around
struct Mdl_sdk_state
{
    mi::base::Handle<mi::neuraylib::INeuray>          mdl_sdk;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>  mdl_impexp_api;
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api;
    mi::base::Handle<mi::neuraylib::IMdl_factory>     mdl_factory;
    mi::base::Handle<mi::neuraylib::ITransaction>     transaction;
};

// Convert to floating point and transform to linear space if necessary
mi::base::Handle<const mi::neuraylib::ICanvas> adjust_canvas(
    mi::neuraylib::IImage_api* image_api,
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas,
    mi::Float32 gamma)
{
    // For simplicity we convert all textures to floating point and pre-apply gamma.
    const char *pixel_type = canvas->get_type();
    const char *target_type;
    if (strcmp(pixel_type, "Rgba") == 0 ||
        strcmp(pixel_type, "Rgbea") == 0 ||
        strcmp(pixel_type, "Rgba_16") == 0 ||
        strcmp(pixel_type, "Color") == 0)
        target_type = "Color";
    else if (
        strcmp(pixel_type, "Sint8") == 0 ||
        strcmp(pixel_type, "Sint32") == 0 ||
        strcmp(pixel_type, "Float32") == 0)
        target_type = "Float32";
    else
        target_type = "Rgb_fp";

    const bool type_conversion = strcmp(pixel_type, target_type) != 0;
    // Nothing to do?
    if (gamma == 1.0f && !type_conversion)
        return canvas;

    if (gamma != 1.0f) {
        // Copy/convert, transform to linear space.
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
            image_api->convert(canvas.get(), target_type));
        gamma_canvas->set_gamma(gamma);
        image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    } else if (type_conversion) {
        // Convert to expected format.
        canvas = image_api->convert(canvas.get(), target_type);
    }

    return canvas;
}

// Loads an image from file, convert to float and transform to linear space
mi::base::Handle<const mi::neuraylib::ICanvas> load_image_from_file(
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::ITransaction* transaction,
    const char* filename)
{
    mi::base::Handle<const mi::neuraylib::ICanvas> result;

    mi::base::Handle<mi::neuraylib::IImage> image(
        transaction->create<mi::neuraylib::IImage>("Image"));
    if (image->reset_file(filename) == 0)
    {
        // poor man's gamma guess
        mi::Float32 gamma = 1.0f;
        std::string t = image->get_type(0, 0);
        if (t == "Rgb" || t == "Rgba")
            gamma = 2.2f;

        result = adjust_canvas(
            image_api, mi::base::make_handle(image->get_canvas(0, 0, 0)), gamma);
    }
    return result;
}

// Returns the canvas holding an image contained in the database
mi::base::Handle<const mi::neuraylib::ICanvas> load_image_from_db(
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::ITransaction* transaction,
    const char* texture_name)
{
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(texture_name));
    if (!texture)
        return mi::base::Handle<const mi::neuraylib::ICanvas>();

    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    if (!image)
        return mi::base::Handle<const mi::neuraylib::ICanvas>();

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));

    if (image->is_uvtile() || image->is_animated()) {
        std::cerr << "The example does not support uvtile and/or animated textures!" << std::endl;
        return mi::base::Handle<const mi::neuraylib::ICanvas>();
    }

    return adjust_canvas(image_api, canvas, texture->get_effective_gamma(0, 0));
}

// Saves the given GL texture to disc
mi::Sint32 save_image(
    const Mdl_sdk_state& state,
    const char* filename,
    GLuint texture,
    GLuint w,
    GLuint h)
{
    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        state.mdl_sdk->get_api_component<mi::neuraylib::IImage_api>());

    mi::base::Handle <mi::neuraylib::ICanvas> canvas(image_api->create_canvas("Rgb_fp", w, h));
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
    void* data = tile->get_data();
    glBindTexture(GL_TEXTURE_2D, texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, data);

    return state.mdl_impexp_api->export_canvas(filename, canvas.get());
}

// Returns a GLSL function string of the given name that returns the given value
std::string get_color_default(const std::string& name, const mi::Color& value = mi::Color(0.0f))
{
    return
        "vec3 " + name + "(State state) {\n"
        "    return vec3(" + to_string(value.r) + +"," +
        to_string(value.g) + ", " + to_string(value.b) + ");\n"
        "}\n";
}

// Returns a GLSL function string of the given name that returns the given value
std::string get_float_default(const std::string& name, mi::Float32 value = 0.0f)
{
    return
        "float " + name + "(State state) {\n"
        "    return float(" + to_string(value) + ");\n"
        "}\n";
}

// Returns a GLSL function string of the given name that returns state::normal
std::string get_normal_default(const std::string& name)
{
    return
        "vec3 " + name + "(State state) {\n"
        "    return state.normal;\n"
        "}\n";
}

// Returns a GLSL function string of the given name that returns a texture at index
std::string get_vector_texture_access(const std::string& name, mi::Size index)
{
    return
        "vec3 " + name + "(State state) {\n"
        "    return texture(mdl_textures_2d[" +
        to_string(index) + "], state.text_coords[0].rg).rgb;\n"
        "}\n";
}

// Returns a GLSL function string of the given name that returns a normal in world space
// read from the texture at index
std::string get_normal_texture_access(const std::string& name, mi::Size index)
{
    return
        "vec3 " + name + "(State state) {\n"
        "    vec3 n = texture(mdl_textures_2d[" +
        to_string(index) + "], state.text_coords[0].rg).rgb;\n"
        "return normalize("
             "state.tangent_u[0] * n.x + "
             "state.tangent_v[0] * n.y + "
             "state.normal * n.z);\n"
        "}\n";
}

// Returns a GLSL function string of the given name that returns a texture at index
std::string get_float_texture_access(const std::string& name, mi::Size index)
{
    return
        "float " + name + "(State state) {\n"
        "    return texture(mdl_textures_2d[" +
        to_string(index) + "], state.text_coords[0].rg).r;\n"
        "}\n";
}

// MDL-2-GLSL utility. Abstract base class, collects relevant material expressions from
// a UE4-distilled MDL material. Overriders decide what to do with those expressions (bake,
// generate code). See Mdl_ue4_GLSL and Mdl_ue4_baker for details.
class Mdl_ue4
{
public:

    virtual ~Mdl_ue4() {}

    // Returns the GLSL code for this material
    virtual const char* get_code() const = 0;

    // Returns the number of textures of the GLSL material
    virtual mi::Size get_texture_count() const = 0;

    // Returns the texture at index
    virtual const mi::neuraylib::ICanvas* get_texture(mi::Size index) const = 0;

    // Returns the number of read-only data segments used by the GLSL material
    virtual mi::Size get_ro_data_segment_count() const
    {
        return 0;
    }

    // Returns the read-only data segment data at index
    virtual const char* get_ro_data_segment_data(mi::Size) const
    {
        return 0;
    }

    // Returns the size of the read-only data segment data at index
    virtual mi::Size get_ro_data_segment_size(mi::Size) const
    {
        return 0;
    }

    // Returns the name of the read-only data segment data at index
    virtual const char* get_ro_data_segment_name(mi::Size) const
    {
        return 0;
    }

protected:

    // add material expression for parameter parameter_name at path
    virtual mi::Sint32 add_material_expression(
        const std::string& parameter_name, const std::string& path) = 0;

    virtual void add_color_default(
        const std::string& parameter_name, const mi::Color& value = mi::Color(0.0f)) = 0;

    virtual void add_float_default(
        const std::string& parameter_name, const mi::Float32& value = mi::Float32(0.0f)) = 0;

    virtual void add_normal_default(const std::string& parameter_name) = 0;

    // Traverses the distilled material and calls add_material_expression for each
    // expression relevant for the UE4 material model
    void add_ue4_material_expressions(
        mi::neuraylib::ITransaction* transaction, const mi::neuraylib::ICompiled_material* cm)
    {
        // Access surface.scattering function
        mi::base::Handle<const mi::neuraylib::IExpression_direct_call> parent_call(
            lookup_call("surface.scattering", cm));
        // ... and get its semantic
        mi::neuraylib::IFunction_definition::Semantics semantic(
            get_call_semantic(transaction, parent_call.get()));

        bool has_clearcoat = false;
        bool has_metallic = false;
        bool is_metallic = false;
        bool has_base_color = false;
        bool has_specular = false;
        bool has_roughness = false;
        bool has_normal = false;
        std::string path_prefix = "surface.scattering.";

        // Check for a clearcoat layer, first. If present, it is the outermost layer
        if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        {
            // Add clearcoat expression paths
            add_material_expression("clearcoat_weight", path_prefix + "weight");
            add_material_expression("clearcoat_color", path_prefix + "layer.tint");
            add_material_expression("clearcoat_roughness", path_prefix + "layer.roughness_u");
            add_material_expression("clearcoat_normal", path_prefix + "normal");

            has_clearcoat = true;

            // Get clear-coat base layer
            parent_call = lookup_call("base", cm, parent_call.get());
            // Get clear-coat base layer semantic
            semantic = get_call_semantic(transaction, parent_call.get());
            // Extend path prefix
            path_prefix += "base.";
        }
        // Check for a weighted layer. Sole purpose of this layer is the transportation of
        // the under-clearcoat-normal. It contains an empty base and a layer with the
        // actual material body
        if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_WEIGHTED_LAYER)
        {
            // Collect under-clearcoat normal
            add_material_expression("normal", path_prefix + "normal");
            has_normal = true;

            // Chain further
            parent_call = lookup_call("layer", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "layer.";
        }
        // Check for a normalized mix. This mix combines the metallic and dielectric parts
        // of the material
        if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_NORMALIZED_MIX)
        {
            // The top-mix component is supposed to be a glossy bsdf
            // Collect metallic weight
            add_material_expression("metallic", path_prefix + "components.value1.weight");
            add_material_expression("roughness",
                path_prefix + "components.value1.component.roughness_u");
            add_material_expression("base_color", path_prefix + "components.value1.component.tint");

            has_roughness = true;
            has_metallic = true;
            has_base_color = true;

            // Chain further
            parent_call = lookup_call(
                "components.value0.component", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "components.value0.component.";
        }
        if (semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        {
            // Collect specular parameters
            add_material_expression("specular", path_prefix + "weight");
            has_specular = true;

            if (!has_roughness)
            {
                add_material_expression("roughness", path_prefix + "layer.roughness_u");
                has_roughness = true;
            }

            // Chain further
            parent_call = lookup_call("base", cm, parent_call.get());
            semantic = get_call_semantic(transaction, parent_call.get());
            path_prefix += "base.";
        }
        if (semantic ==
            mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
        {
            if (!has_metallic)
            {
                if (!has_base_color)
                {
                    add_material_expression("base_color", path_prefix + "tint");
                    has_base_color = true;
                }

                if (!has_roughness)
                {
                    add_material_expression("roughness", path_prefix + "roughness_u");
                    has_roughness = true;
                }
                is_metallic = true;
            }
        }
        else if (semantic ==
            mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF)
        {
            if (!has_base_color)
            {
                add_material_expression("base_color", path_prefix + "tint");
                has_base_color = true;
            }
        }
        if (!has_normal)
            add_material_expression("normal", "geometry.normal");


        if (!has_base_color) // should not happen
            add_color_default("base_color");

        if (!has_roughness)
            add_float_default("roughness");
        if (!has_metallic)
            add_float_default("metallic", is_metallic ? 1.0f : 0.0f);
        if (!has_specular)
            add_float_default("specular");
        if (!has_clearcoat)
        {
            add_color_default("clearcoat_color");
            add_float_default("clearcoat_weight");
            add_float_default("clearcoat_roughness");
            add_normal_default("clearcoat_normal");
        }
    }

    mi::base::Handle<const mi::neuraylib::ICompiled_material> m_cm;
};

// Specialization of Mdl_ue4 which generates GLSL code for all relevant material
// expressions.
class Mdl_ue4_glsl : public Mdl_ue4
{
public:

    Mdl_ue4_glsl(const Mdl_sdk_state& state, const mi::neuraylib::ICompiled_material* cm)
        : m_result(0)
        , m_cm(mi::base::make_handle_dup(cm))
        , m_transaction(state.transaction)
        , m_image_api(state.mdl_sdk->get_api_component<mi::neuraylib::IImage_api>())
        , m_context(state.mdl_factory->create_execution_context())
    {
        // Access GLSL backend
        m_be_glsl = state.mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_GLSL);
        check_success(m_be_glsl);

        // Set backend options
        check_success(m_be_glsl->set_option("num_texture_spaces", "1") == 0);

#ifdef USE_SSBO
        // SSBO requires GLSL 4.30
        check_success(m_be_glsl->set_option("glsl_version", "430") == 0);
#else
        check_success(m_be_glsl->set_option("glsl_version", "330") == 0);
#endif

        check_success(m_be_glsl->set_option("glsl_state_normal_mode", "field") == 0);
        check_success(m_be_glsl->set_option("glsl_state_position_mode", "field") == 0);
        check_success(m_be_glsl->set_option("glsl_state_texture_coordinate_mode", "field") == 0);
        check_success(m_be_glsl->set_option("glsl_state_texture_tangent_u_mode", "field") == 0);
        check_success(m_be_glsl->set_option("glsl_state_texture_tangent_v_mode", "field") == 0);
        check_success(m_be_glsl->set_option("glsl_state_texture_space_max_mode", "field") == 0);

#ifdef USE_SSBO
        check_success(m_be_glsl->set_option("glsl_max_const_data", "0") == 0);
        check_success(m_be_glsl->set_option("glsl_place_uniforms_into_ssbo", "on") == 0);
#else
        check_success(m_be_glsl->set_option("glsl_max_const_data", "1024") == 0);
        check_success(m_be_glsl->set_option("glsl_place_uniforms_into_ssbo", "off") == 0);
#endif

#ifdef REMAP_NOISE_FUNCTIONS
        // remap noise functions that access the constant tables
        check_success(m_be_glsl->set_option("glsl_remap_functions",
            "_ZN4base12perlin_noiseEu6float4=noise_float4"
            ",_ZN4base12worley_noiseEu6float3fi=noise_worley"
            ",_ZN4base8mi_noiseEu6float3=noise_mi_float3"
            ",_ZN4base8mi_noiseEu4int3=noise_mi_int3") == 0);
#endif

        // Create link unit
        m_link_unit = m_be_glsl->create_link_unit(m_transaction.get(), m_context.get());

        add_ue4_material_expressions(m_transaction.get(), m_cm.get());

        // Generate code
        m_target_code = m_be_glsl->translate_link_unit(m_link_unit.get(), m_context.get());
        check_success(print_messages(m_context.get()));

        m_code = m_target_code->get_code();
#ifdef REMAP_NOISE_FUNCTIONS
        m_code += read_text_file(get_executable_folder() + "/" + "noise_no_lut.glsl");
#endif
        m_code += m_defaults;
        add_texture_runtime();
    }

    virtual ~Mdl_ue4_glsl()
    {
    }

    virtual void add_color_default(
        const std::string& parameter_name, const mi::Color& value = mi::Color(0.0f))
    {
        m_defaults += get_color_default("get_" + parameter_name, value);
    }

    virtual void add_float_default(
        const std::string& parameter_name, const mi::Float32& value = mi::Float32(0.0f))
    {
        m_defaults += get_float_default("get_" + parameter_name, value);
    }

    virtual void add_normal_default(const std::string& parameter_name)
    {
        m_defaults += get_normal_default("get_" + parameter_name);
    }

    virtual mi::Sint32 add_material_expression(
        const std::string& parameter_name, const std::string& path)
    {
        const std::string fct_name = "get_" + parameter_name;
        mi::Sint32 result = m_link_unit->add_material_expression(
            m_cm.get(), path.c_str(), fct_name.c_str(), m_context.get());
        print_messages(m_context.get());

        return result;
    }

    virtual const char* get_code() const
    {
        return m_code.c_str();
    }

    virtual mi::Size get_texture_count() const
    {
        return m_target_code->get_texture_count();
    }

    virtual const mi::neuraylib::ICanvas* get_texture(mi::Size index) const
    {

        mi::base::Handle<const mi::neuraylib::ICanvas> canvas =
            load_image_from_db(
                m_image_api.get(), m_transaction.get(), m_target_code->get_texture(index));

        if (!canvas)
            return nullptr;
        canvas->retain();
        return canvas.get();
    }

    virtual mi::Size get_ro_data_segment_count() const
    {
        return m_target_code->get_ro_data_segment_count();
    }

    virtual const char* get_ro_data_segment_data(mi::Size index) const
    {
        return m_target_code->get_ro_data_segment_data(index);
    }

    virtual mi::Size get_ro_data_segment_size(mi::Size index) const
    {
        return m_target_code->get_ro_data_segment_size(index);
    }

    virtual const char* get_ro_data_segment_name(mi::Size index) const
    {
        return m_target_code->get_ro_data_segment_name(index);
    }

private:

    mi::Sint32 m_result;
    std::string m_defaults;
    std::string m_code;

    mi::base::Handle <const mi::neuraylib::ICompiled_material>
                                                        m_cm;
    mi::base::Handle <mi::neuraylib::ITransaction>      m_transaction;
    mi::base::Handle<mi::neuraylib::IImage_api>         m_image_api;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context>
                                                        m_context;

    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
    mi::base::Handle<mi::neuraylib::ILink_unit>         m_link_unit;
    mi::base::Handle<mi::neuraylib::IMdl_backend>       m_be_glsl;

    // Create texture runtime
    void add_texture_runtime()
    {
        if (m_target_code->get_texture_count() == 0)
            return;

        m_code += "\n";
        m_code += "#define MAX_TEXTURES " + to_string(
            m_target_code->get_texture_count() - 1) + "\n";
        m_code += "uniform sampler2D mdl_textures_2d[MAX_TEXTURES];\n";
        m_code +=
            "int tex_width_2d(int tex, ivec2 uv_tile, float frame) {\n"
            "    if (tex == 0) return 0; // invalid texture\n"
            "    if (tex > MAX_TEXTURES) return 0;\n"
            "    return textureSize(mdl_textures_2d[tex-1], 0).x;\n"
            "}\n"
            "int tex_height_2d(int tex, ivec2 uv_tile, float frame) {\n"
            "    if (tex == 0) return 0; // invalid texture\n"
            "    if (tex > MAX_TEXTURES) return 0;\n"
            "    return textureSize(mdl_textures_2d[tex-1], 0).y;\n"
            "}\n"
            "vec3 tex_lookup_float3_2d("
            "int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)\n"
            "{\n"
            "    if (tex == 0) return vec3(0);\n"
            "    if (tex > MAX_TEXTURES) return vec3(0);\n"
            "    return texture(mdl_textures_2d[tex-1], coord).rgb;\n"
            "}\n"
            "vec3 tex_lookup_color_2d("
            "int tex, vec2 coord, int wrap_u, int wrap_v, vec2 crop_u, vec2 crop_v, float frame)\n"
            "{\n"
            "    if (tex == 0) return vec3(0);\n"
            "    if (tex > MAX_TEXTURES) return vec3(0);\n"
            "    return texture(mdl_textures_2d[tex-1], coord).rgb;\n"
            "}\n"
            "vec3 tex_texel_color_2d(int tex, ivec2 coord, ivec2 uv_tile, float frame)\n"
            "{\n"
            "    if (tex == 0) return vec3(0);\n"
            "    if (tex > MAX_TEXTURES) return vec3(0);\n"
            "    return texelFetch(mdl_textures_2d[tex-1], coord, 0).rgb;\n"
            "}\n";

        std::string w_switch, h_switch;

        for (mi::Size i = 1; i < m_target_code->get_texture_count(); ++i)
        {
            mi::base::Handle<const mi::neuraylib::ITexture> tex(
                m_transaction->access<const mi::neuraylib::ITexture>(
                    m_target_code->get_texture(i)));
            mi::base::Handle<const mi::neuraylib::IImage> image(
                m_transaction->access<const mi::neuraylib::IImage>(tex->get_image()));
            w_switch += "    case " + to_string(i) + "u: return "
                + to_string(image->resolution_x(0, 0, 0)) + ";\n";
            h_switch += "    case " + to_string(i) + "u: return "
                + to_string(image->resolution_y(0, 0, 0)) + ";\n";
        }
        m_code +=
            "int tex_width(uint tex, ivec2 uv_tile, float frame)"
            "{\n"
            "    switch (tex) {\n"
            "    case 0u: return 0;\n"
            + w_switch +
            "    }\n"
            "}\n"
            "int tex_height(uint tex, ivec2 uv_tile, float frame)"
            "{\n"
            "    switch (tex) {\n"
            "    case 0u: return 0;\n"
            + h_switch +
            "    }\n"
            "}\n";
        return;
    }
};

// Helper struct that describes a material parameter for the target shader
struct Material_parameter
{
    mi::Size canvas_index;
    mi::base::Handle<mi::IData> value;
    std::string pixel_type;
    std::string bake_path;

    Material_parameter(const std::string& pixel_type)
        : canvas_index(~0u), pixel_type(pixel_type) { }

    Material_parameter()
        : canvas_index(~0u) { }
};

// Specialization of Mdl_ue4 which bakes all relevant material
// expressions into textures or constants
class Mdl_ue4_baker : public Mdl_ue4
{
public:

    Mdl_ue4_baker(
        const Mdl_sdk_state& state,
        const mi::neuraylib::ICompiled_material* cm,
        int baking_resolution_x, int baking_resolution_y)
        : m_sdk_state(state)
        , m_cm(mi::base::make_handle_dup(cm))
        , m_baking_resolution_x(baking_resolution_x)
        , m_baking_resolution_y(baking_resolution_y)
    {
        // initialize material parameters
        m_material_parameters["base_color"]          = Material_parameter("Rgb_fp");
        m_material_parameters["metallic"]            = Material_parameter("Float32");
        m_material_parameters["specular"]            = Material_parameter("Float32");
        m_material_parameters["roughness"]           = Material_parameter("Float32");
        m_material_parameters["normal"]              = Material_parameter("Float32<3>");

        m_material_parameters["clearcoat_color"]     = Material_parameter("Rgb_fp");
        m_material_parameters["clearcoat_weight"]    = Material_parameter("Float32");
        m_material_parameters["clearcoat_roughness"] = Material_parameter("Float32");
        m_material_parameters["clearcoat_normal"]    = Material_parameter("Float32<3>");

        // collect parameters from cm
        add_ue4_material_expressions(m_sdk_state.transaction.get(), cm);

        // bake
        bake_expressions();

        // generate access code
        generate_glsl();
    }

    virtual ~Mdl_ue4_baker() { }

    virtual void add_color_default(
        const std::string& parameter_name, const mi::Color& value = mi::Color(0.0f))
    {
        Material_parameter_map::iterator param = m_material_parameters.find(parameter_name);
        if (param == m_material_parameters.end())
            return;

        param->second.value = create_value(m_sdk_state.transaction.get(), "Color", value);
    }

    virtual void add_float_default(
        const std::string& parameter_name, const mi::Float32& value = mi::Float32(0.0f))
    {
        Material_parameter_map::iterator param = m_material_parameters.find(parameter_name);
        if (param == m_material_parameters.end())
            return;

        param->second.value = create_value(m_sdk_state.transaction.get(), "Float32", value);
    }

    virtual void add_normal_default(const std::string& parameter_name)
    {
        Material_parameter_map::iterator param = m_material_parameters.find(parameter_name);
        if (param == m_material_parameters.end())
            return;

        param->second.bake_path = "geometry.normal";
    }

    virtual mi::Sint32 add_material_expression(
        const std::string& parameter_name, const std::string& path)
    {

        Material_parameter_map::iterator param = m_material_parameters.find(parameter_name);
        if (param == m_material_parameters.end())
            return -1;

        param->second.bake_path = path;
        return 0;
    }

    virtual const char* get_code() const
    {
        return m_code.c_str();
    }

    virtual mi::Size get_texture_count() const
    {
        return m_textures.size() ? m_textures.size() + 1 : 0;
    }

    virtual const mi::neuraylib::ICanvas* get_texture(mi::Size index) const
    {
        if (index < 1 || index > m_textures.size())
            return nullptr;

        m_textures[index-1]->retain();
        return m_textures[index-1].get();
    }


private:

    std::string m_code;
    typedef std::map<std::string, Material_parameter> Material_parameter_map;

    std::map<std::string, Material_parameter> m_material_parameters;
    std::vector <mi::base::Handle<mi::neuraylib::ICanvas> > m_textures;

    Mdl_sdk_state m_sdk_state;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> m_cm;

    int m_baking_resolution_x;
    int m_baking_resolution_y;

    // bake expressions to texture
    void bake_expressions()
    {
        mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
            m_sdk_state.mdl_sdk->get_api_component<mi::neuraylib::IMdl_distiller_api>());

        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            m_sdk_state.mdl_sdk->get_api_component<mi::neuraylib::IImage_api>());

        for (Material_parameter_map::iterator it = m_material_parameters.begin();
            it != m_material_parameters.end(); ++it )
        {
            Material_parameter& param = it->second;

            // Do not attempt to bake empty paths
            if (param.bake_path.empty())
                continue;

            // Create baker for current path
            mi::base::Handle<const mi::neuraylib::IBaker> baker(distiller_api->create_baker(
                m_cm.get(), param.bake_path.c_str(), mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK));
            check_success(baker.is_valid_interface());

            if (baker->is_uniform())
            {
                mi::base::Handle<mi::IData> value;
                if (param.pixel_type == "Rgb_fp")
                    param.value = create_value(
                        m_sdk_state.transaction.get(), "Color", mi::Color(0.f));
                else if (param.pixel_type == "Float32")
                    param.value = create_value(m_sdk_state.transaction.get(), "Float32", 0.f);
                else if (param.pixel_type == "Float32<3>")
                    param.value = create_value(
                        m_sdk_state.transaction.get(), "Float32<3>", mi::Float32_3(0.f));
                else
                {
                    std::cout << "Ignoring unsupported value type '" << param.pixel_type
                        << "'" << std::endl;
                    continue;
                }

                // Bake constant value
                mi::Sint32 result = baker->bake_constant(param.value.get());
                check_success(result == 0);
            }
            else
            {
                // Create a canvas
                mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                    image_api->create_canvas(param.pixel_type.c_str(),
                        m_baking_resolution_x, m_baking_resolution_y));

                // Bake texture
                mi::Sint32 result = baker->bake_texture(canvas.get(), 1);
                check_success(result == 0);

                m_textures.push_back(canvas);
                param.canvas_index = m_textures.size() - 1;
            }
        }
    }

    void generate_glsl()
    {
        m_code +=
            "#version 330 core\n"
            "struct State {\n"
            "    vec3 normal;\n"
            "    vec3 geom_normal;\n"
            "    vec3 position;\n"
            "    float animation_time;\n"
            "    vec3 text_coords[1];\n"
            "    vec3 tangent_u[1];\n"
            "    vec3 tangent_v[1];\n"
            "    int ro_data_segment_offset;\n"
            "    mat4 world_to_object;\n"
            "    mat4 object_to_world;\n"
            "    int object_id;\n"
            "    float meters_per_scene_unit;\n"
            "    int arg_block_offset;\n"
            "};\n\n";

        if (m_textures.size())
        {
            m_code += "#define MAX_TEXTURES " + to_string(
                m_textures.size()) + "\n";
            m_code += "uniform sampler2D mdl_textures_2d[MAX_TEXTURES];\n";
        }

        for (Material_parameter_map::iterator it = m_material_parameters.begin();
            it != m_material_parameters.end(); ++it)
        {
            Material_parameter& param = it->second;
            if (param.canvas_index == ~0u)
            {
                if (param.pixel_type == "Rgb_fp")
                {
                    mi::Color v;
                    mi::get_value(param.value.get(), v);
                    m_code += get_color_default("get_" + it->first, v);
                }
                else if (param.pixel_type == "Float32")
                {
                    mi::Float32 v;
                    mi::get_value(param.value.get(), v);
                    m_code += get_float_default("get_" + it->first, v);
                }
                else if (param.pixel_type == "Float32<3>")
                {
                    m_code += get_normal_default("get_" + it->first);
                }
                else
                    check_success(!"unsupported pixel type");
            }
            else
            {
                if (param.pixel_type == "Rgb_fp")
                {
                    m_code += get_vector_texture_access("get_" + it->first, param.canvas_index);
                }
                else if (param.pixel_type == "Float32")
                {
                    m_code += get_float_texture_access("get_" + it->first, param.canvas_index);
                }
                else if (param.pixel_type == "Float32<3>")
                {
                    m_code += get_normal_texture_access("get_" + it->first, param.canvas_index);
                }
                else
                    check_success(!"unsupported pixel type");
            }
        }
        m_code += "\n";
    }
};

//------------------------------------------------------------------------------
//
// OpenGL code
//
//------------------------------------------------------------------------------

// Struct representing a vertex of a scene object.
struct Vertex {
    mi::Float32_3 position;
    mi::Float32_3 normal;
    mi::Float32_3 tangent;
    mi::Float32_3 binormal;
    mi::Float32_2 tex_coord;
    Vertex(
        const mi::Float32_3& p,
        const mi::Float32_3& n,
        const mi::Float32_3& t,
        const mi::Float32_3& b,
        const mi::Float32_2& uv)
        : position(p), normal(n), tangent(t), binormal(b), tex_coord(uv)
    { }
};

// Error callback for GLFW.
static void handle_glfw_error(int error_code, const char* description)
{
    std::cerr << "GLFW error (code: " << error_code << "): \"" << description << "\"\n";
}

// Initialize OpenGL and create a window with an associated OpenGL context.
static GLFWwindow *init_opengl(unsigned int w, unsigned int h, bool show_window)
{
    printf("Setting GLFW err callback ...\n");
    glfwSetErrorCallback(handle_glfw_error);

    printf("Initializing GLFW ...\n");
    // Initialize GLFW
    check_success(glfwInit());

#ifdef USE_SSBO
    printf("Setting GLSL 4.3 version hint ...\n");
    // SSBO requires GLSL 4.30
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#else
    printf("Setting GLSL 3.3 version hint ...\n");
    // else GLSL 3.30 is sufficient
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
    printf("Setting OpenGL profile hint ...\n");
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    printf("Setting OpenGL forward compatibility hint ...\n");
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    printf("Setting window visibility hint ...\n");
    glfwWindowHint(GLFW_VISIBLE, show_window);
    // Create an OpenGL window and a context
    printf("Creating GLFW window ...\n");
    GLFWwindow *window = glfwCreateWindow(
        w, h, "MDL Distilling Example", nullptr, nullptr);
    if (!window) {
        std::cerr << "Error creating OpenGL window!" << std::endl;
        terminate();
    }
    printf("Attach context to window ...\n");
    // Attach context to window
    glfwMakeContextCurrent(window);

  // Initialize GLEW to get OpenGL extensions
    printf("Initializing GLEW ...\n");
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        std::cerr << "GLEW error: " << glewGetErrorString(res) << std::endl;
        terminate();
    }

    printf("Enabling depth test ...\n");
    glEnable(GL_DEPTH_TEST);
    printf("Setting texture cube map seamless ...\n");
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // Enable VSync
    printf("Enable vsync ...\n");
    glfwSwapInterval(1);

    printf("Checking for OpenGL errors ...\n");
    check_gl_success();

    return window;
}

// Create a vertex array
static void create_vertex_array(
    GLuint& vao, GLuint& vbo, const GLvoid* vertices, GLsizei vertices_size)
{
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
}

// Create an index buffer
static void create_index_buffer(GLuint& ebo, const GLvoid* indices, GLsizei indices_size)
{
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
}

// Create an OpenGL texture
static GLuint create_gl_texture(GLenum target, GLsizei w, GLsizei h, GLenum format=GL_RGB,
    const void* data=nullptr,
    GLenum min_filter = GL_LINEAR,
    GLenum mag_filter = GL_LINEAR,
    GLenum wrap_s = GL_CLAMP_TO_EDGE,
    GLenum wrap_t = GL_CLAMP_TO_EDGE,
    GLenum wrap_r = GL_CLAMP_TO_EDGE,
    int levels = 0)
{
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(target, id);

    GLenum int_format;
    switch (format) {
        case GL_RED:
            int_format = GL_R32F;
            break;
        case GL_RGB:
            int_format = GL_RGB32F;
            break;
        default:
        case GL_RGBA:
            int_format = GL_RGBA32F;
            break;
    }

    if(target == GL_TEXTURE_CUBE_MAP)
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, int_format,
                w, h, 0, format, GL_FLOAT, data);
        }
    else {
        if (levels)
        {
            glTexStorage2D(GL_TEXTURE_2D, levels, int_format, w, h);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, format, GL_FLOAT, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
            glTexImage2D(target, 0, int_format,
                w, h, 0, format, GL_FLOAT, data);
    }
    glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap_t);
    if (target == GL_TEXTURE_CUBE_MAP)
        glTexParameteri(target, GL_TEXTURE_WRAP_R, wrap_r);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter);

    check_gl_success();

    return id;
}

static GLenum get_gl_format(const char *pixel_type)
{
    if (strcmp(pixel_type, "Float32") == 0)
        return GL_RED;
    else if (strcmp(pixel_type, "Color") == 0)
        return GL_RGBA;
    else
        return GL_RGB;
}

// Convert the given canvas to an OpenGL texture
static GLuint load_gl_texture(
    GLenum target, mi::base::Handle<const mi::neuraylib::ICanvas> canvas, int levels = 0)
{
    if (!canvas)
        return static_cast<GLuint>(-1);

    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
    const GLenum format = get_gl_format(canvas->get_type());

    return create_gl_texture(target, canvas->get_resolution_x(), canvas->get_resolution_y(),
        format,
        tile->get_data(),
        levels ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR,
        GL_LINEAR, GL_REPEAT, GL_REPEAT, GL_REPEAT, levels);
}

// Create the shader program with a fragment shader
static GLuint create_shader_program(
    const std::string& vertex_program,
    const std::string& fragment_program)
{
    GLint success;

    GLuint program = glCreateProgram();

    add_shader(GL_VERTEX_SHADER, vertex_program, program);
    add_shader(GL_FRAGMENT_SHADER, fragment_program, program);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        dump_program_info(program, "Error linking the shader program: ");
        terminate();
    }

    check_gl_success();

    return program;
}

// Class wrapping an OpenGL shader program
class Shader_program
{
public:

    Shader_program(const std::string& vertex_shader_file, const std::string& fragment_shader_file)
    {
        m_program = create_shader_program(
            read_text_file(vertex_shader_file),
            read_text_file(fragment_shader_file));
    }

    Shader_program() : m_program(-1) {
    }

    virtual ~Shader_program()
    {
        glDeleteProgram(m_program);
    }

    virtual void bind_textures() const
    {

    }

    void make_current() const
    {
        glUseProgram(m_program);
    }

    void set_float(const char* argument, mi::Float32 value)
    {
        GLint location = glGetUniformLocation(m_program, argument);
        if (location >= 0)
            glUniform1f(location, value);
    }

    void set_int(const char* argument, mi::Uint32 value)
    {
        GLint location = glGetUniformLocation(m_program, argument);
        if (location >= 0)
            glUniform1i(location, value);
    }

    void set_vector3(const char* argument, const mi::Float32_3& value)
    {
        GLint location = glGetUniformLocation(m_program, argument);
        if (location >= 0)
            glUniform3fv(location, 1, value.begin());
    }

    void set_matrix(const char* argument, const mi::Float32_4_4& value)
    {
        GLint location = glGetUniformLocation(m_program, argument);
        if(location >= 0)
            glUniformMatrix4fv(location, 1, GL_FALSE, value.begin());
    }

    void set_vertex_attrib_float(const char* arg, GLsizei size, GLsizei stride, GLsizei offset)
    {
        GLint location = glGetAttribLocation(m_program, arg);
        if (location < 0)
            return;
        glEnableVertexAttribArray(location);
        glVertexAttribPointer(
            location, size, GL_FLOAT, GL_FALSE, stride,
            reinterpret_cast<const void*>(size_t(offset)));
    }

protected:

    GLuint m_program;

private:
    Shader_program(const Shader_program& other) = delete;

    Shader_program& operator=(const Shader_program&) = delete;

};

// Our U4-like PBR shader
class Mdl_pbr_shader : public Shader_program
{
public:
    // Constructor
    Mdl_pbr_shader(
        const Mdl_sdk_state& state,
        Mdl_ue4* mdl_ue4,
        GLuint irradiance_map,
        GLuint refl_map,
        GLuint brdf_lut_map)

        : m_irradiance_map(irradiance_map)
        , m_refl_map(refl_map)
        , m_brdf_lut_map(brdf_lut_map)
    {
        // Setup GLSL programs
        // Get fragment code generated from MDL expressions
        std::string fragment_code = mdl_ue4->get_code();
        // Add main fragment shader
        fragment_code += read_text_file(
            mi::examples::io::get_executable_folder() + "/" + "example_distilling_glsl.frag");

#ifdef DUMP_GLSL
        std::fstream file;
        file.open("glsl_dump.frag", std::fstream::out);
        file << fragment_code;
        file.close();
#endif
        // Compile and link shaders
        m_program = create_shader_program(read_text_file(
            mi::examples::io::get_executable_folder() + "/" + "example_distilling_glsl.vert"),
            fragment_code);

        // Assign texture slots for IBL maps
        make_current();
        set_int("irradiance_map", 0);
        set_int("refl_map", 1);
        set_int("brdf_lut", 2);

        // Upload RO data
        set_mdl_readonly_data(mdl_ue4);

        // Upload all textures referenced by the target code (the compiled material expressions)
        // to the gpu and store their handles

        // Skip invalid texture (always at index 0)
        for (mi::Size i = 1; i < mdl_ue4->get_texture_count(); ++i)
        {
            m_mdl_textures.push_back(
                load_gl_texture(GL_TEXTURE_2D,
                    mi::base::make_handle<const mi::neuraylib::ICanvas>(mdl_ue4->get_texture(i))));
            // first 3 indices are reserved for IBL maps
            std::string name = "mdl_textures_2d[" + to_string(i-1) + "]";
            set_int(name.c_str(), mi::Uint32(i + 2));
        }

        delete mdl_ue4;
    }

    // Destructor
    virtual ~Mdl_pbr_shader()
    {
        if (m_buffer_objects.size() > 0)
            glDeleteBuffers(GLsizei(m_buffer_objects.size()), &m_buffer_objects[0]);

        if (m_mdl_textures.size() > 0)
            glDeleteTextures(GLsizei(m_mdl_textures.size()), &m_mdl_textures[0]);

        check_gl_success();
    }

    virtual void bind_textures() const
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_irradiance_map);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_refl_map);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, m_brdf_lut_map);

        for (size_t i=0; i<m_mdl_textures.size(); ++i)
        {
            // first 3 indices are reserved for IBL maps
            glActiveTexture(GLenum(GL_TEXTURE0 + i + 3));
            glBindTexture(GL_TEXTURE_2D, m_mdl_textures[i]);
        }
    }

private:
    // Sets the read-only data segments in the current OpenGL program object.
    void set_mdl_readonly_data(
        const Mdl_ue4* mdl_ue4)
    {
        mi::Size num_uniforms = mdl_ue4->get_ro_data_segment_count();
        if (num_uniforms == 0)
            return;

#ifdef USE_SSBO
        GLuint next_storage_block_binding = 0u;
        m_buffer_objects.resize(num_uniforms);

        glGenBuffers(GLsizei(num_uniforms), &m_buffer_objects[0]);

        for (mi::Size i = 0; i < num_uniforms; ++i) {
            mi::Size segment_size = mdl_ue4->get_ro_data_segment_size(i);
            char const* segment_data = mdl_ue4->get_ro_data_segment_data(i);

#ifdef DUMP_GLSL
            std::cout << "Dump ro segment data " << i << " \""
                << mdl_ue4->get_ro_data_segment_name(i) << "\" (size = "
                << segment_size << "):\n" << std::hex;

            for (int j = 0; j < 16 && j < segment_size; ++j) {
                std::cout << "0x" << (unsigned int)(unsigned char) segment_data[j] << ", ";
            }
            std::cout << std::dec << std::endl;
#endif

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_buffer_objects[i]);
            glBufferData(
                GL_SHADER_STORAGE_BUFFER, GLsizeiptr(segment_size), segment_data, GL_STATIC_DRAW);

            GLuint block_index = glGetProgramResourceIndex(
                m_program, GL_SHADER_STORAGE_BLOCK,
                mdl_ue4->get_ro_data_segment_name(i));
            glShaderStorageBlockBinding(m_program, block_index, next_storage_block_binding);
            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                next_storage_block_binding,
                m_buffer_objects[i]);

            ++next_storage_block_binding;

            check_gl_success();
        }
#else
        std::vector<char const*> uniform_names;
        for (mi::Size i = 0; i < num_uniforms; ++i) {
#ifdef DUMP_GLSL
            mi::Size segment_size = mdl_ue4->get_ro_data_segment_size(i);
            const char* segment_data = mdl_ue4->get_ro_data_segment_data(i);

            std::cout << "Dump ro segment data " << i << " \""
                << mdl_ue4->get_ro_data_segment_name(i) << "\" (size = "
                << segment_size << "):\n" << std::hex;

            for (int i = 0; i < 16 && i < segment_size; ++i) {
                std::cout << "0x" << (unsigned int)(unsigned char)segment_data[i] << ", ";
            }
            std::cout << std::dec << std::endl;
#endif

            uniform_names.push_back(mdl_ue4->get_ro_data_segment_name(i));
        }

        std::vector<GLuint> uniform_indices(num_uniforms, 0);
        glGetUniformIndices(
            m_program, GLsizei(num_uniforms), &uniform_names[0], &uniform_indices[0]);

        for (mi::Size i = 0; i < num_uniforms; ++i) {
            // uniforms may have been removed, if they were not used
            if (uniform_indices[i] == GL_INVALID_INDEX)
                continue;

            GLint uniform_type = 0;
            GLuint index = GLuint(uniform_indices[i]);
            glGetActiveUniformsiv(m_program, 1, &index, GL_UNIFORM_TYPE, &uniform_type);

#ifdef DUMP_GLSL
            std::cout << "Uniform type of " << uniform_names[i]
                << ": 0x" << std::hex << uniform_type << std::dec << std::endl;
#endif

            mi::Size segment_size = mdl_ue4->get_ro_data_segment_size(i);
            const char* segment_data = mdl_ue4->get_ro_data_segment_data(i);

            GLint uniform_location = glGetUniformLocation(m_program, uniform_names[i]);

            switch (uniform_type) {

                // For bool, the data has to be converted to int, first
#define CASE_TYPE_BOOL(type, func, num)                            \
    case type: {                                                   \
        GLint *buf = new GLint[segment_size];                      \
        for (mi::Size j = 0; j < segment_size; ++j)                \
            buf[j] = GLint(segment_data[j]);                       \
        func(uniform_location, GLsizei(segment_size / num), buf);  \
        delete[] buf;                                              \
        break;                                                     \
    }

                CASE_TYPE_BOOL(GL_BOOL, glUniform1iv, 1)
                CASE_TYPE_BOOL(GL_BOOL_VEC2, glUniform2iv, 2)
                CASE_TYPE_BOOL(GL_BOOL_VEC3, glUniform3iv, 3)
                CASE_TYPE_BOOL(GL_BOOL_VEC4, glUniform4iv, 4)

#define CASE_TYPE(type, func, num, elemtype)                                      \
    case type:                                                                    \
        func(uniform_location, GLsizei(segment_size / (num * sizeof(elemtype))),  \
            (const elemtype*)segment_data);                                       \
        break

                CASE_TYPE(GL_INT, glUniform1iv, 1, GLint);
                CASE_TYPE(GL_INT_VEC2, glUniform2iv, 2, GLint);
                CASE_TYPE(GL_INT_VEC3, glUniform3iv, 3, GLint);
                CASE_TYPE(GL_INT_VEC4, glUniform4iv, 4, GLint);
                CASE_TYPE(GL_FLOAT, glUniform1fv, 1, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC2, glUniform2fv, 2, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC3, glUniform3fv, 3, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC4, glUniform4fv, 4, GLfloat);
                CASE_TYPE(GL_DOUBLE, glUniform1dv, 1, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC2, glUniform2dv, 2, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC3, glUniform3dv, 3, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC4, glUniform4dv, 4, GLdouble);

#define CASE_TYPE_MAT(type, func, num, elemtype)                                  \
    case type:                                                                    \
        func(uniform_location, GLsizei(segment_size / (num * sizeof(elemtype))),  \
            false, (const elemtype*)segment_data);                                \
        break

                CASE_TYPE_MAT(GL_FLOAT_MAT2_ARB, glUniformMatrix2fv, 4, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT2x3, glUniformMatrix2x3fv, 6, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3x2, glUniformMatrix3x2fv, 6, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT2x4, glUniformMatrix2x4fv, 8, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4x2, glUniformMatrix4x2fv, 8, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3_ARB, glUniformMatrix3fv, 9, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3x4, glUniformMatrix3x4fv, 12, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4x3, glUniformMatrix4x3fv, 12, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4_ARB, glUniformMatrix4fv, 16, GLfloat);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2, glUniformMatrix2dv, 4, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2x3, glUniformMatrix2x3dv, 6, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3x2, glUniformMatrix3x2dv, 6, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2x4, glUniformMatrix2x4dv, 8, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4x2, glUniformMatrix4x2dv, 8, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3, glUniformMatrix3dv, 9, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3x4, glUniformMatrix3x4dv, 12, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4x3, glUniformMatrix4x3dv, 12, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4, glUniformMatrix4dv, 16, GLdouble);

            default:
                std::cerr << "Unsupported uniform type: 0x"
                    << std::hex << uniform_type << std::dec << std::endl;
                terminate();
                break;
            }

            check_gl_success();
        }
#endif
    }

private:

    // IBL maps
    GLuint m_irradiance_map;
    GLuint m_refl_map;
    GLuint m_brdf_lut_map;

    // mdl expression textures
    std::vector<GLuint> m_mdl_textures;

    std::vector<GLuint> m_buffer_objects;
};

// Opengl Mesh base class
class Mesh
{
public:

    virtual void draw() = 0;

    virtual ~Mesh() {}

    virtual void bind_shader(Shader_program*)
    {

    }
};

// Simple sphere
class Sphere : public Mesh
{
public:

    Sphere(const float radius, const unsigned int slices, const unsigned int stacks)
    {
        std::vector<unsigned int> indices;
        std::vector<Vertex> vertices;

        const float step_phi = (float) (2.0 * M_PI / slices);
        const float step_theta = (float) (M_PI / stacks);

        for (unsigned int i = 0; i <= stacks; ++i) {

            const float theta = step_theta * float(i);
            const float sin_t = sin(theta);
            const float cos_t = cos(theta);

            for (unsigned int j = 0; j <= slices; ++j) {

                const float phi = step_phi * float(j);
                const float sin_p = sin(phi);
                const float cos_p = cos(phi);

                const mi::Float32_3 p(sin_p * sin_t, cos_t, cos_p * sin_t);
                const mi::Float32_2 uv(
                    2.0f * float(j) / float(slices), 1.0f - float(i) / float(stacks));

                mi::Float32_3 tangent_u = mi::Float32_3(cos_p, 0.0f, -sin_p);
                mi::Float32_3 tangent_v = cross(p, tangent_u);

                vertices.push_back(Vertex(p * radius, p, tangent_u, tangent_v, uv));
            }
        }

        for (unsigned int i = 0; i < stacks; ++i) {
            for (unsigned int j = 0; j < slices; ++j) {

                const unsigned int p0 = i * (slices + 1) + j;
                const unsigned int p1 = p0 + 1;
                const unsigned int p2 = p0 + slices + 1;
                const unsigned int p3 = p2 + 1;

                indices.push_back(p0);
                indices.push_back(p1);
                indices.push_back(p2);

                indices.push_back(p1);
                indices.push_back(p3);
                indices.push_back(p2);
            }
        }

        create_vertex_array(m_vao, m_vbo, vertices.data(),
            (GLsizei) (vertices.size() * sizeof(Vertex)));


        create_index_buffer(m_ebo, indices.data(), (GLsizei) (
            indices.size() * sizeof(unsigned int)));

        m_nindices = (GLuint) indices.size();

        check_gl_success();
    }

    virtual ~Sphere()
    {
        // cleanup
        glDeleteVertexArrays(1, &m_vao);
        glDeleteBuffers(1, &m_vbo);
        glDeleteBuffers(1, &m_ebo);
    }

    virtual void bind_shader(Shader_program* program)
    {
        // set locations of vertex shader inputs

        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

        program->make_current();
        program->set_vertex_attrib_float(
            "Position", 3, sizeof(Vertex), 0);
        program->set_vertex_attrib_float(
            "Normal", 3, sizeof(Vertex), sizeof(mi::Float32_3));
        program->set_vertex_attrib_float(
            "Tangent", 3, sizeof(Vertex), sizeof(mi::Float32_3) * 2);
        program->set_vertex_attrib_float(
            "Binormal", 3, sizeof(Vertex), sizeof(mi::Float32_3) * 3);
        program->set_vertex_attrib_float(
            "TexCoord", 2, sizeof(Vertex), sizeof(mi::Float32_3) * 4);
    }

    virtual void draw()
    {
        glBindVertexArray(m_vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);

        glDrawElements(
            GL_TRIANGLES,      // mode
            m_nindices,        // count
            GL_UNSIGNED_INT,   // type
            (void*)0           // element array buffer offset
        );
    }

private:

    GLuint m_vbo, m_vao, m_ebo;
    GLuint m_nindices;
};

// Screen aligned quad
class Screen_aligned_quad : public Mesh
{
public:
    Screen_aligned_quad()
    {
        static const float vertices1[] =
        {
            -1.f, -1.f, 1.0f,
             1.f, -1.f, 1.0f,
            -1.f,  1.f, 1.0f,
             1.f, -1.f, 1.0f,
             1.f,  1.f, 1.0f,
            -1.f,  1.f, 1.0f
        };
        create_vertex_array(m_vao, m_vbo, vertices1, sizeof(vertices1));

        check_gl_success();
    }

    virtual void bind_shader(Shader_program* program)
    {
        program->make_current();
        program->set_vertex_attrib_float("Position", 3, sizeof(float) * 3, 0);
    }

    virtual void draw()
    {
        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glEnable(GL_DEPTH_TEST);
    }

   virtual  ~Screen_aligned_quad()
    {
        // cleanup
        glDeleteVertexArrays(1, &m_vao);
        glDeleteBuffers(1, &m_vbo);
    }

private:
    GLuint m_vao;
    GLuint m_vbo;
};

// Convert degree to radians
static mi::Float64 deg_to_rad(mi::Float64 v)
{
    return v * 3.1415926 / 180.;
}

// returns a perspective projection matrix
static mi::Float32_4_4 projection(
    const mi::Float32 fovy,
    const mi::Float32 aspect_ratio,
    const mi::Float32 near_plane,
    const mi::Float32 far_plane
)
{
    const mi::Float32
        y_scale = (mi::Float32) (1.0 / tan(deg_to_rad(fovy * 0.5))),
        x_scale = y_scale / aspect_ratio,
        frustum_length = far_plane - near_plane;

    mi::Float32_4_4 p(0.f);
    p.xx = x_scale;
    p.yy = y_scale;
    p.zz = -((far_plane + near_plane) / frustum_length);
    p.wz = -((2 * near_plane * far_plane) / frustum_length);
    p.zw = -1;

    return p;
}

// Generates the prefiltered glossy map as a cubemap with mipmaps for increasing roughness, see
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
static GLuint prefilter_glossy(GLsizei w, GLsizei h, GLuint env_tex_id, GLuint env_accel_tex_id)
{
    unsigned int fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    unsigned int cubemap_id = create_gl_texture(
        GL_TEXTURE_CUBE_MAP, w, h, GL_RGB, nullptr, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    Shader_program program(
        mi::examples::io::get_executable_folder() + "/" + "screen_aligned_quad.vert",
        mi::examples::io::get_executable_folder() + "/" + "prefilter_glossy.frag");
    Screen_aligned_quad quad;
    quad.bind_shader(&program);

    mi::Float32_4_4 proj = projection(90.f, 1.0, 0.1f, 10.f);
    proj.invert();
    program.make_current();
    program.set_matrix("inv_proj", proj);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, env_tex_id);
    program.set_int("env_tex", 0);

    if (env_accel_tex_id) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, env_accel_tex_id);
        program.set_int("env_accel_tex", 1);
    }

    mi::Float32_4_4 mv[6];
    const mi::Float32_3 pos(0.0f, 0.0f, 0.0f);
    mv[0].lookat(pos, mi::Float32_3( 1.0f,  0.0f,  0.0f),  mi::Float32_3(0.0f, -1.0f,  0.0f));
    mv[1].lookat(pos, mi::Float32_3(-1.0f,  0.0f,  0.0f),  mi::Float32_3(0.0f, -1.0f,  0.0f));
    mv[2].lookat(pos, mi::Float32_3( 0.0f,  1.0f,  0.0f),  mi::Float32_3(0.0f,  0.0f,  1.0f));
    mv[3].lookat(pos, mi::Float32_3( 0.0f, -1.0f,  0.0f),  mi::Float32_3(0.0f,  0.0f, -1.0f));
    mv[4].lookat(pos, mi::Float32_3( 0.0f,  0.0f,  1.0f),  mi::Float32_3(0.0f, -1.0f,  0.0f));
    mv[5].lookat(pos, mi::Float32_3( 0.0f,  0.0f, -1.0f),  mi::Float32_3(0.0f, -1.0f,  0.0f));

    for (unsigned int i = 0; i < 6; ++i)
        mv[i].transpose();

    const unsigned int miplevel_count = 5;
    for (unsigned int mip = 0; mip < miplevel_count; ++mip)
    {
        const unsigned int mip_w = (unsigned int) (w * std::pow(0.5, mip));
        const unsigned int mip_h = (unsigned int) (h * std::pow(0.5, mip));

        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mip_w, mip_h);
        glViewport(0, 0, mip_w, mip_h);

        float roughness = (float) mip / (float) (miplevel_count - 1);
        program.set_float("roughness", roughness);

        for (unsigned int i = 0; i < 6; ++i)
        {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, cubemap_id, mip);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            program.set_matrix("inv_mv", mv[i]);
            quad.draw();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return cubemap_id;
}

// generates a diffuse irradiance map
static GLuint prefilter_diffuse(GLsizei w, GLsizei h, GLuint env_tex_id, GLuint env_accel_tex_id)
{
    unsigned int fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    unsigned int cubemap_id = create_gl_texture(GL_TEXTURE_CUBE_MAP, w, h);

    mi::Float32_4_4 mv[6], proj = projection(90.f, 1.0, 0.1f, 10.f);
    proj.invert();

    const mi::Float32_3 pos(0.0f, 0.0f, 0.0f);
    mv[0].lookat(pos, mi::Float32_3(1.0f, 0.0f, 0.0f), mi::Float32_3(0.0f, -1.0f, 0.0f));
    mv[1].lookat(pos, mi::Float32_3(-1.0f, 0.0f, 0.0f), mi::Float32_3(0.0f, -1.0f, 0.0f));
    mv[2].lookat(pos, mi::Float32_3(0.0f, 1.0f, 0.0f), mi::Float32_3(0.0f, 0.0f, 1.0f));
    mv[3].lookat(pos, mi::Float32_3(0.0f, -1.0f, 0.0f), mi::Float32_3(0.0f, 0.0f, -1.0f));
    mv[4].lookat(pos, mi::Float32_3(0.0f, 0.0f, 1.0f), mi::Float32_3(0.0f, -1.0f, 0.0f));
    mv[5].lookat(pos, mi::Float32_3(0.0f, 0.0f, -1.0f), mi::Float32_3(0.0f, -1.0f, 0.0f));

    Shader_program program(
        mi::examples::io::get_executable_folder() + "/" + "screen_aligned_quad.vert",
        mi::examples::io::get_executable_folder() + "/" + "prefilter_diffuse.frag");
    Screen_aligned_quad quad;
    quad.bind_shader(&program);

    program.make_current();
    program.set_matrix("inv_proj", proj);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, env_tex_id);
    program.set_int("env_tex", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, env_accel_tex_id);
    program.set_int("env_accel_tex", 1);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, cubemap_id, 0);
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        mv[i].transpose();
        program.set_matrix("inv_mv", mv[i]);
        quad.draw();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return cubemap_id;
}

// Create an off-screen render target
GLuint create_offscreen_render_target(GLsizei w, GLsizei h, GLuint& out_fb, GLuint& out_rb)
{
    // Create and setup texture
    unsigned int tex_id = create_gl_texture(GL_TEXTURE_2D, w, h);

    // Create and bind frame buffer and render buffer objects
    glGenFramebuffers(1, &out_fb);
    glGenRenderbuffers(1, &out_rb);

    glBindFramebuffer(GL_FRAMEBUFFER, out_fb);
    glBindRenderbuffer(GL_RENDERBUFFER, out_rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, out_rb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_id, 0);

    return tex_id;
}

// Pre-integrate glossy brdf, see
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
GLuint integrate_brdf(GLsizei w, GLsizei h)
{
    Shader_program program(
        mi::examples::io::get_executable_folder() + "/" + "integrate_brdf.vert",
        mi::examples::io::get_executable_folder() + "/" + "integrate_brdf.frag");
    Screen_aligned_quad quad;
    GLuint fbo=0, rbo=0;
    GLuint tex_id = create_offscreen_render_target(w, h, fbo, rbo);
    // Render screen aligned quad
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    quad.bind_shader(&program);
    quad.draw();
    // reset framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);
    return tex_id;
}

//------------------------------------------------------------------------------
//
// Application logic
//
//------------------------------------------------------------------------------

// Context structure for window callback functions.
struct Window_context
{
    unsigned int width, height;
    bool         moving;
    double       move_start_x, move_start_y;
    double       move_dx, move_dy;
    int          zoom;
    int          material;
    bool         bake;
    float        exposure;
    bool         event;

    Window_context()
        : width(1024), height(1024)
        , moving(false)
        , move_start_x(0.f), move_start_y(0.f)
        , move_dx(0.f), move_dy(0.f)
        , zoom(0)
        , material(0)
        , bake(true)
        , exposure(0.0f)
        , event(false)
    {
    }
};

// GLFW scroll callback
static void handle_scroll(GLFWwindow *window, double /*xoffset*/, double yoffset)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0.0) {
        ctx->zoom++;
        ctx->event = true;
    }
    else if (yoffset < 0.0) {
        ctx->zoom--;
        ctx->event = true;
    }
}

// GLFW callback handler for keyboard inputs.
void handle_key(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    // Handle key press events
    if (action == GLFW_PRESS) {

        switch (key) {
            // Escape closes the window
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_RIGHT:
                ++ctx->material;
                ctx->event = true;
                break;
            case GLFW_KEY_LEFT:
                --ctx->material;
                ctx->event = true;
                break;
            case GLFW_KEY_UP:
            case GLFW_KEY_DOWN:
                ctx->bake = !ctx->bake;
                ctx->event = true;
                break;
            case GLFW_KEY_KP_SUBTRACT:
                ctx->exposure --;
                ctx->event = true;
                break;
            case GLFW_KEY_KP_ADD:
                ctx->exposure ++;
                ctx->event = true;
                break;
            default:
                break;
        }
    }
}

// GLFW mouse button callback
static void handle_mouse_button(GLFWwindow *window, int button, int action, int /*mods*/)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            ctx->moving = true;
            glfwGetCursorPos(window, &ctx->move_start_x, &ctx->move_start_y);
        }
        else
            ctx->moving = false;
    }
}

// GLFW mouse position callback
static void handle_mouse_pos(GLFWwindow *window, double xpos, double ypos)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (ctx->moving)
    {
        ctx->move_dx += xpos - ctx->move_start_x;
        ctx->move_dy += ypos - ctx->move_start_y;
        ctx->move_start_x = xpos;
        ctx->move_start_y = ypos;
        ctx->event = true;
    }
}

// GLFW callback handler for framebuffer resize events (when window size or resolution changes).
void handle_framebuffer_size(GLFWwindow* window, int width, int height)
{
    Window_context *ctx = static_cast<Window_context*>(
        glfwGetWindowUserPointer(window));
    ctx->width = width;
    ctx->height = height;
    ctx->event = true;

    glViewport(0, 0, width, height);
}

struct Env_accel {
    unsigned int alias;
    float q;
    float pdf;
};

// Build alias map
static float build_alias_map(
    const float *data,
    const unsigned int size,
    Env_accel *accel)
{
    // create qs (normalized)
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; ++i)
        sum += data[i];

    for (unsigned int i = 0; i < size; ++i)
        accel[i].q = (static_cast<float>(size) * data[i] / sum);

    // create partition table
    unsigned int *partition_table = static_cast<unsigned int *>(
        malloc(size * sizeof(unsigned int)));
    unsigned int s = 0u, large = size;
    for (unsigned int i = 0; i < size; ++i)
        partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] = accel[i].alias = i;

    // create alias map
    for (s = 0; s < large && large < size; ++s)
    {
        const unsigned int j = partition_table[s], k = partition_table[large];
        accel[j].alias = k;
        accel[k].q += accel[j].q - 1.0f;
        large = (accel[k].q < 1.0f) ? (large + 1u) : large;
    }

    free(partition_table);

    return sum;
}

// Create environment map texture and acceleration data for importance sampling
static GLuint create_environment_accel_texture(
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas)
{

    const mi::Uint32 rx = canvas->get_resolution_x();
    const mi::Uint32 ry = canvas->get_resolution_y();

    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
    const float *pixels = static_cast<const float *>(tile->get_data());

    // Create importance sampling data
    Env_accel *env_accel = static_cast<Env_accel *>(malloc(rx * ry * sizeof(Env_accel)));
    float *importance_data = static_cast<float *>(malloc(rx * ry * sizeof(float)));
    float cos_theta0 = 1.0f;
    const float step_phi = float(2.0 * M_PI) / float(rx);
    const float step_theta = float(M_PI) / float(ry);
    for (unsigned int y = 0; y < ry; ++y)
    {
        const float theta1 = float(y + 1) * step_theta;
        const float cos_theta1 = std::cos(theta1);
        const float area = (cos_theta0 - cos_theta1) * step_phi;
        cos_theta0 = cos_theta1;

        for (unsigned int x = 0; x < rx; ++x) {
            const unsigned int idx = y * rx + x;
            const unsigned int idx3 =  idx * 3;
            importance_data[idx] =
                area * std::max(pixels[idx3], std::max(pixels[idx3 + 1], pixels[idx3 + 2]));
        }
    }
    const float inv_env_integral = 1.0f / build_alias_map(importance_data, rx * ry, env_accel);
    free(importance_data);
    for (unsigned int i = 0; i < rx * ry; ++i) {
        const unsigned int idx3 = i * 3;
        env_accel[i].pdf =
            std::max(pixels[idx3], std::max(pixels[idx3 + 1], pixels[idx3 + 2])) * inv_env_integral;
    }
    GLuint tex_id = create_gl_texture(GL_TEXTURE_2D, rx, ry,
        GL_RGB, env_accel, GL_NEAREST, GL_NEAREST);

    free(env_accel);

    return tex_id;
}

// Initializes OpenGL, creates the shader program, sets up the scene
// and shows the window/renders to file.
void render_scene(
    const Mdl_sdk_state& state,
    const std::vector<mi::base::Handle<const mi::neuraylib::ICompiled_material> >
     & distilled_materials,
    const Options& options)
{
    check_success(distilled_materials.size());

    Window_context window_context;
    window_context.bake = options.bake;
    window_context.exposure = options.exposure;

    float exposure_scale = static_cast<float>(pow(2.0, options.exposure));

    // Init OpenGL window and setup event callbacks

    printf("Initializing OpenGL ...\n");
    GLFWwindow *window = init_opengl(
        window_context.width, window_context.height, options.show_window);

    if (options.show_window)
    {
        printf("Setting window user pointer ...\n");
        glfwSetWindowUserPointer(window, &window_context);
        printf("Setting key callback ...\n");
        glfwSetKeyCallback(window, handle_key);
        printf("Setting framebuffer size callback ...\n");
        glfwSetFramebufferSizeCallback(window, handle_framebuffer_size);
        printf("Setting scroll callback ...\n");
        glfwSetScrollCallback(window, handle_scroll);
        printf("Setting cursor position callback ...\n");
        glfwSetCursorPosCallback(window, handle_mouse_pos);
        printf("Setting mouse button callback ...\n");
        glfwSetMouseButtonCallback(window, handle_mouse_button);
    }
    printf("Initializing OpenGL done.\n");

    // Get image API
    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        state.mdl_sdk->get_api_component<mi::neuraylib::IImage_api>());

    // Load environment texture and compute IBL maps
    printf("Generating maps ...\n");

    mi::base::Handle<const mi::neuraylib::ICanvas> env_tex_canvas =
        load_image_from_file(image_api.get(), state.transaction.get(), options.hdrfile.c_str());

    GLuint env_accel_tex_id = create_environment_accel_texture(env_tex_canvas);

    GLuint env_tex_id = load_gl_texture(GL_TEXTURE_2D, env_tex_canvas);

    GLuint iblmap_id = prefilter_diffuse(128, 128, env_tex_id, env_accel_tex_id);
    GLuint refmap_id = prefilter_glossy(512, 512, env_tex_id, env_accel_tex_id);
    GLuint brdflutid = integrate_brdf(512, 512);

    env_tex_canvas = 0;
    glDeleteTextures(1, &env_accel_tex_id);

    printf("Generating maps done.\n");

    glViewport(0, 0, window_context.width, window_context.height);
    {
        // Create scene data
        std::vector<Mdl_pbr_shader*> pbr_shaders(distilled_materials.size() * 2);

        std::cout << "Generating shader " << window_context.material <<
            " in " << (window_context.bake ? "baked" : "GLSL")
            << " mode ..." << std::endl;

        int index = window_context.bake ? 0 : 1;

        Mdl_ue4* mdl_ue4 = 0;
        if (window_context.bake)
            mdl_ue4 = new Mdl_ue4_baker(state, distilled_materials[0].get(),
                options.baking_resolution_x, options.baking_resolution_y);
        else
            mdl_ue4 = new Mdl_ue4_glsl(state, distilled_materials[0].get());

        Mdl_pbr_shader* sphere_shader = new Mdl_pbr_shader(
            state, mdl_ue4,
            iblmap_id, refmap_id, brdflutid);
        pbr_shaders[0+index] = sphere_shader;

        std::cout << "Generating shader done." << std::endl;

        Sphere sphere(1.f, 64, 64);
        sphere.bind_shader(sphere_shader);

        Shader_program env_shader(
            mi::examples::io::get_executable_folder() + "/" + "screen_aligned_quad.vert",
            mi::examples::io::get_executable_folder() + "/" + "environment_sphere.frag");
        Screen_aligned_quad quad;
        quad.bind_shader(&env_shader);

        // Camera position
        float cam_dist = 3.f;
        double phi = 0.0f, theta = (M_PI * 0.5);
        mi::Float32_3 cam_pos(0.f, 0.f, cam_dist);
        mi::Float32_3 cam_interest(0.f, 0.f, 0.f);
        mi::Float32_3 cam_up(0.f, 1.f, 0.f);

        mi::Float32_4_4 mv(1.0f);
        mv.lookat(cam_pos, cam_interest, cam_up);
        mi::Float32_4_4 inv_mv(mv);
        inv_mv.transpose();

        mi::Float32_4_4 proj =
            projection(
                45.f, float(window_context.width) / float(window_context.height), 1.f, 100.f);
        mi::Float32_4_4 inv_proj(proj);
        inv_proj.invert();

        // Loop until the user closes the window
        if(options.show_window)
            while (!glfwWindowShouldClose(window))
            {
                // Render the scene

                if (window_context.event)
                {
                    const int num_materials = int(distilled_materials.size());
                    window_context.material = window_context.material % num_materials;
                    if (window_context.material < 0)
                        window_context.material += num_materials;

                    index = window_context.bake ? 0 : 1;
                    sphere_shader =
                        pbr_shaders[window_context.material * 2 + index];
                    // create, if it does not exist yet
                    if (!sphere_shader)
                    {
                        std::cout << "Generating shader " << window_context.material <<
                            " in " << (index ? "GLSL" : "baked" )
                            << " mode ..." << std::endl;

                        if (window_context.bake)
                            mdl_ue4 = new Mdl_ue4_baker(
                                state, distilled_materials[window_context.material].get(),
                                options.baking_resolution_x, options.baking_resolution_y);
                        else
                            mdl_ue4 = new Mdl_ue4_glsl(
                                state, distilled_materials[window_context.material].get());

                        sphere_shader = new Mdl_pbr_shader(
                            state, mdl_ue4, iblmap_id, refmap_id, brdflutid);

                        pbr_shaders[window_context.material * 2 + index] = sphere_shader;

                        std::cout << "Generating shader done." << std::endl;
                    }
                    sphere.bind_shader(sphere_shader);

                    exposure_scale = static_cast<float>(pow(2.0, window_context.exposure));

                    phi -= window_context.move_dx * 0.001 * M_PI;
                    theta -= window_context.move_dy * 0.001 * M_PI;
                    theta = std::max(theta, 0.05 * M_PI);
                    theta = std::min(theta, 0.95 * M_PI);
                    window_context.move_dx = window_context.move_dy = 0.0;

                    cam_pos.x = float(sin(phi) * sin(theta));
                    cam_pos.y = float(cos(theta));
                    cam_pos.z = float(cos(phi) * sin(theta));

                    cam_up.x = float(-sin(phi) * cos(theta));
                    cam_up.y = float(sin(theta));
                    cam_up.z = float(-cos(phi) * cos(theta));

                    const float dist = float(cam_dist * pow(0.95, double(window_context.zoom)));
                    cam_pos *= dist;
                    mv.lookat(cam_pos, cam_interest, cam_up);
                    inv_mv = mv;
                    inv_mv.transpose();

                    proj = projection(45.f,
                        float(window_context.width) / float(window_context.height), 1.f, 100.f);
                    inv_proj = proj;
                    inv_proj.invert();

                    window_context.event = false;
                }

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                env_shader.make_current();
                env_shader.set_float("exposure_scale", exposure_scale);
                env_shader.set_matrix("inv_mv", inv_mv);
                env_shader.set_matrix("inv_proj", inv_proj);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, env_tex_id);

                quad.draw();

                sphere_shader->make_current();
                sphere_shader->set_float("exposure_scale", exposure_scale);
                sphere_shader->set_matrix("m_view", mv);
                sphere_shader->set_matrix("m_projection", proj);
                sphere_shader->set_vector3("cam_position", cam_pos);
                sphere_shader->bind_textures();

                sphere.draw();

                // Swap front and back buffers
                glfwSwapBuffers(window);

                // Poll for events and process them
                glfwPollEvents();
            }
        else
        {
            // render to texture
            GLuint fbo = 0, rbo = 0;
            GLuint fb = create_offscreen_render_target(
                window_context.width, window_context.height, fbo, rbo);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            env_shader.make_current();
            env_shader.set_float("exposure_scale", exposure_scale);
            env_shader.set_matrix("inv_mv", inv_mv);
            env_shader.set_matrix("inv_proj", inv_proj);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, env_tex_id);

            quad.draw();

            sphere_shader->make_current();
            sphere_shader->set_float("exposure_scale", exposure_scale);
            sphere_shader->set_matrix("m_view", mv);
            sphere_shader->set_matrix("m_projection", proj);
            sphere_shader->set_vector3("cam_position", cam_pos);
            sphere_shader->bind_textures();

            sphere.draw();

            save_image(state, options.outputfile.c_str(), fb,
                window_context.width, window_context.height);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        for (mi::Size i = 0; i < pbr_shaders.size(); ++i)
            delete pbr_shaders[i];
    }

    glDeleteTextures(1, &env_tex_id);
    glDeleteTextures(1, &refmap_id);
    glDeleteTextures(1, &iblmap_id);
    glDeleteTextures(1, &brdflutid);

    check_gl_success();
    glfwDestroyWindow(window);
    glfwTerminate();
}

// Distill material to given target model
const mi::neuraylib::ICompiled_material* distill_material(
    const Mdl_sdk_state& state,
    const std::string& target_model,
    mi::neuraylib::ICompiled_material* cm)
{
    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        state.mdl_sdk->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    check_success(distiller_api);
    const mi::neuraylib::ICompiled_material* dm =
        distiller_api->distill_material(cm, target_model.c_str());
    check_success(dm);
    return dm;
}

// Compile material
mi::neuraylib::ICompiled_material* compile_material(
    const Mdl_sdk_state& state,
    const std::string& material_name)
{
    // Create execution context
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        state.mdl_factory->create_execution_context());

    // split module and material name
    std::string module_name, material_simple_name;
    if (!mi::examples::mdl::parse_cmd_argument_material_name(
        material_name, module_name, material_simple_name, true))
            return nullptr;

    // Load the module.
    state.mdl_impexp_api->load_module(state.transaction.get(), module_name.c_str(), context.get());
    if (!print_messages(context.get()))
        return nullptr;

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        state.mdl_factory->get_db_module_name(module_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        state.transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // Attach the material name
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name);
    if (material_db_name.empty())
        exit_failure("Failed to find the material %s in the module %s.",
            material_simple_name.c_str(), module_name.c_str());

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        state.transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        return nullptr;

   // Create instance
   mi::base::Handle<mi::neuraylib::IFunction_call> mat_inst(
       material_definition->create_function_call(nullptr));
   if (!mat_inst)
       return nullptr;

   // Compile
   mi::base::Handle<mi::neuraylib::IMaterial_instance> mat_inst2(
       mat_inst->get_interface<mi::neuraylib::IMaterial_instance>());
   mi::base::Handle<mi::neuraylib::ICompiled_material> cm(
       mat_inst2->create_compiled_material(
           mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS, context.get()));
   if (!print_messages(context.get()))
       return nullptr;

   if (!cm)
       return nullptr;
   cm->retain();
   return cm.get();
}

// Prints program usage
static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "-h                       print this text\n"
        << "--nowin                  don't open interactive display\n"
        << "--hdr <filename>         HDR environment map (default: textures/environment.hdr)\n"
        << "-o <outputfile>          image file to write result to (default: output.exr)\n"
        << "-e <exposure>            exposure for interactive display (default: 0.0)\n"
        << "--no_baking              do not bake UE4 material parameters to textures but generate"
                                     " GLSL code\n"
        << "-r <w> <h>               baking resolution (default: 2048x2048)\n"
        << "--mdl_path <path>        mdl search path, can occur multiple times.\n";

    exit(EXIT_FAILURE);
}

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line
    Options options;
    mi::examples::mdl::Configure_options configure_options;

    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--nowin") == 0) {
                options.show_window = false;
            } else if (strcmp(opt, "--mdl_path") == 0 && i < argc - 1) {
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            } else if (strcmp(opt, "--hdr") == 0 && i < argc - 1) {
                options.hdrfile = argv[++i];
            } else if (strcmp(opt, "--no_baking") == 0) {
                options.bake = false;
            } else if (strcmp(opt, "-e") == 0 && i < argc - 1) {
                options.exposure = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "-o") == 0 && i < argc - 1) {
                options.outputfile = argv[++i];
            } else if (strcmp(opt, "-r") == 0 && i < argc - 2) {
                options.baking_resolution_x = atoi(argv[++i]);
                options.baking_resolution_y = atoi(argv[++i]);
            } else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        }
        else
            options.material_names.push_back(opt);
    }
    if(options.material_names.empty())
        options.material_names.push_back(
            "::nvidia::sdk_examples::tutorials_distilling::example_distilling2");

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Load the distilling plugin
    if (mi::examples::mdl::load_plugin(neuray.get(), "mdl_distiller" MI_BASE_DLL_FILE_EXT) != 0)
        exit_failure("Failed to load the mdl_distiller plugin.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        //  Access required API components
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
        mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
            neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

        Mdl_sdk_state sdk_state;
        sdk_state.mdl_sdk = neuray;
        sdk_state.mdl_impexp_api = mdl_impexp_api;
        sdk_state.mdl_backend_api = mdl_backend_api;
        sdk_state.mdl_factory = mdl_factory;

        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        sdk_state.transaction = scope->create_transaction();

        {
            std::vector<mi::base::Handle<const mi::neuraylib::ICompiled_material> >
                distilled_materials;

            for (mi::Size i = 0; i < options.material_names.size(); ++i)
            {
                // Load and compile material
                mi::base::Handle<mi::neuraylib::ICompiled_material> cm(
                    compile_material(
                        sdk_state,
                        options.material_names[i]));
                check_success(cm.is_valid_interface());

                printf("Distilling material %s ...\n", options.material_names[i].c_str());

                // Distill to UE4 target
                mi::base::Handle<const mi::neuraylib::ICompiled_material> dm(
                    distill_material(sdk_state, "ue4", cm.get()));
                check_success(dm.is_valid_interface());

                printf("Distilling material %s ... done.\n", options.material_names[i].c_str());

                distilled_materials.push_back(dm);
            }
            render_scene(
                sdk_state, distilled_materials, options);
        }
        sdk_state.transaction->commit();
    }

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
