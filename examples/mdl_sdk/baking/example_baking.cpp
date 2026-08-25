/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

// examples/mdl_sdk/baking/example_baking.cpp
//
// Bakes selected sub-expressions of an MDL material into textures or constants
// without running the distiller. The set of expressions is either supplied via
// repeatable --expr flags, or, when none is given, taken from a curated default
// list (see g_default_expressions below).

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "example_shared.h"
#include "example_shared_dump.h"
#include "utils/profiling.h"
using namespace mi::examples::profiling;

// Frame/uvtile map populated by build_canvases() for animated / uvtile textures.
// Uvtile is a (u, v) tile origin; std::pair's lexicographic operator< is sufficient.
// This approach assumes that there are no u/v transformations, otherwise it ends up
// baking the wrong u/v pairs.
using Uvtile        = std::pair<mi::Sint32, mi::Sint32>;
using Uvtiles       = std::set<Uvtile>;
using Frame_number  = mi::Size;
using Frame_uvtiles = std::map<Frame_number, Uvtiles>;


// One expression to bake (either from the default list or supplied via --expr).
// `path` is the bake path passed to IMdl_distiller_api::create_baker().
// The sanitized path (dots replaced by underscores) is used as the filename suffix.
struct Bake_expression
{
    std::string path;
};

// Curated default set of bake paths used when --expr is not supplied.
// Paths that do not resolve on the compiled material are skipped at runtime.
const std::vector<Bake_expression> g_default_expressions = {
    { "surface.scattering.tint" },
    { "surface.scattering.roughness" },
    { "geometry.normal" },
    { "geometry.cutout_opacity" },
    { "surface.scattering.roughness_u" },
    { "ior" },
};

// Prints a centred title padded to `width` characters with `fill` on both sides.
void print_header(const std::string& title, int width = 80, char fill = '-')
{
    int padding = width - static_cast<int>(title.size());
    int left = padding / 2;
    int right = padding - left;
    std::cout << std::string(left, fill) << title << std::string(right, fill) << '\n';
}

// Return the file extension to use when saving a baked canvas.
// All bake targets are saved as PNG.
// Note: float - typed canvases(Rgb_fp, Float32, Float32<3> etc.) will now be quantised
// to 8 - bit by the PNG encoder when exported.If any of those baked values fall 
// outside[0, 1] — like HDR emission intensities — they'll be clamped.
const char* canvas_extension(const std::string& /*pixel_type*/)
{
    return ".png";
}

// Returns true if a type can be baked into a texture or constant.
// Based on the type table documented in IBaker (see imdl_distiller_api.h):
//   TK_BOOL, TK_FLOAT, TK_COLOR are always bakeable.
//   TK_VECTOR is bakeable only when the element type is TK_FLOAT and N is 2, 3, or 4.
//   TK_INT, TK_DOUBLE, TK_MATRIX and all other types are not supported by the baker.
bool is_bakeable_type(const mi::neuraylib::IType* type)
{
    switch (type->get_kind()) {
        case mi::neuraylib::IType::TK_BOOL:
        case mi::neuraylib::IType::TK_FLOAT:
        case mi::neuraylib::IType::TK_COLOR:
            return true;
        case mi::neuraylib::IType::TK_VECTOR: {
            mi::base::Handle<const mi::neuraylib::IType_vector> vec(
                type->get_interface<mi::neuraylib::IType_vector>());
            mi::base::Handle<const mi::neuraylib::IType> elem(vec->get_element_type());
            const mi::Size n = vec->get_size();
            return elem->get_kind() == mi::neuraylib::IType::TK_FLOAT
                && n >= 2 && n <= 4;
        }
        default:
            return false;
    }
}

// Recursive helper for collect_bake_paths().
// Walks the expression tree rooted at `expr`, building dot-separated paths as it descends into
// struct fields and BSDF/EDF/VDF constructor arguments. Leaf paths (bakeable types) are appended
// to `out_paths`.
// `visited_temps` prevents re-entering a temporary that has already been fully expanded.
void collect_bake_paths_recursive(
    const mi::neuraylib::IExpression* expr,
    const mi::neuraylib::ICompiled_material* cm,
    const std::string& current_path,
    std::vector<bool>& visited_temps,
    std::vector<std::string>& out_paths)
{
    switch (expr->get_kind()) {

        case mi::neuraylib::IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> direct_call(
                expr->get_interface<mi::neuraylib::IExpression_direct_call>());
            mi::base::Handle<const mi::neuraylib::IType> type(expr->get_type());

            if (!is_bakeable_type(type.get())) {
                // Struct, BSDF, EDF, VDF, hair_bsdf, … – recurse into each named argument.
                // This covers both material struct constructors (material_surface, …) and
                // BSDF/EDF/VDF constructors (diffuse_reflection_bsdf, simple_glossy_bsdf, …)
                // so that paths like surface.scattering.tint are discovered correctly.
                mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                    direct_call->get_arguments());
                for (mi::Size i = 0; i < args->get_size(); ++i) {
                    const char* arg_name = args->get_name(i);
                    mi::base::Handle<const mi::neuraylib::IExpression> arg(
                        args->get_expression(i));
                    std::string child_path = current_path.empty()
                        ? arg_name
                        : current_path + "." + arg_name;
                    collect_bake_paths_recursive(
                        arg.get(), cm, child_path, visited_temps, out_paths);
                }
            } else if (!current_path.empty() && is_bakeable_type(type.get())) {
                // Bakeable direct call (e.g., a texture lookup or math function) – leaf path.
                out_paths.push_back(current_path);
            }
            break;
        }

        case mi::neuraylib::IExpression::EK_CONSTANT:
        case mi::neuraylib::IExpression::EK_PARAMETER: {
            // Constant value or parameter reference – always a leaf.
            if (!current_path.empty()) {
                mi::base::Handle<const mi::neuraylib::IType> type(expr->get_type());
                if (is_bakeable_type(type.get()))
                    out_paths.push_back(current_path);
            }
            break;
        }

        case mi::neuraylib::IExpression::EK_TEMPORARY: {
            mi::base::Handle<const mi::neuraylib::IExpression_temporary> tmp_ref(
                expr->get_interface<mi::neuraylib::IExpression_temporary>());
            mi::Size index = tmp_ref->get_index();

            mi::base::Handle<const mi::neuraylib::IExpression> tmp_expr(
                cm->get_temporary(index));
            mi::base::Handle<const mi::neuraylib::IType> type(tmp_expr->get_type());

            if (!visited_temps[index]) {
                // First encounter: fully expand the temporary under the current path.
                visited_temps[index] = true;
                collect_bake_paths_recursive(
                    tmp_expr.get(), cm, current_path, visited_temps, out_paths);
            } else if (!current_path.empty() && is_bakeable_type(type.get())) {
                // Already expanded from another path: the current reference is still a valid
                // bake path, but we skip re-expanding struct sub-fields to avoid duplication.
                out_paths.push_back(current_path);
            }
            break;
        }

        case mi::neuraylib::IExpression::EK_CALL:
            break; // Indirect calls do not appear in compiled materials.
    }
}

// Enumerates all leaf bake paths present in the compiled material.
// Each returned string is a dot-separated path (e.g., "surface.scattering.tint") that can be
// passed directly to IMdl_distiller_api::create_baker(). Paths whose type cannot be baked
// (BSDF, EDF, VDF, …) are excluded automatically.
std::vector<std::string> collect_bake_paths(
    const mi::neuraylib::ICompiled_material* cm)
{
    std::vector<std::string> paths;
    std::vector<bool> visited_temps(cm->get_temporary_count(), false);

    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(cm->get_body());
    mi::base::Handle<const mi::neuraylib::IExpression_list> args(body->get_arguments());

    for (mi::Size i = 0; i < args->get_size(); ++i) {
        const char* arg_name = args->get_name(i);
        mi::base::Handle<const mi::neuraylib::IExpression> arg(args->get_expression(i));
        collect_bake_paths_recursive(arg.get(), cm, arg_name, visited_temps, paths);
    }

    return paths;
}

// Replace '.' with '_' so a path can be used as a filename suffix.
std::string sanitize_for_filename(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        out.push_back(c == '.' ? '_' : c);
    return out;
}

// Creates an instance of the given material.
mi::neuraylib::IFunction_call* create_material_instance(
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_execution_context* context,
    const std::string& module_qualified_name,
    const std::string& material_simple_name)
{
    // Load the module.
    mdl_impexp_api->load_module(transaction, module_qualified_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_qualified_name.c_str());

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_module_name(module_qualified_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // Attach the material name
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name);
    if (material_db_name.empty())
        exit_failure("Failed to find the material %s in the module %s.",
            material_simple_name.c_str(), module_qualified_name.c_str());

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Accessing definition '%s' failed.", material_db_name.c_str());

    // Create a material instance from the material definition with the default arguments.
    mi::Sint32 result;
    mi::neuraylib::IFunction_call* material_instance =
        material_definition->create_function_call(0, &result);
    if (result != 0)
        exit_failure("Instantiating '%s' failed.", material_db_name.c_str());

    return material_instance;
}

// Compiles the given material instance in the given compilation mode.
mi::neuraylib::ICompiled_material* compile_material_instance(
    mi::neuraylib::IMdl_factory *mdl_factory,
    mi::neuraylib::ITransaction *transaction,
    const mi::neuraylib::IFunction_call* material_instance,
    mi::neuraylib::IMdl_execution_context* context,
    bool class_compilation)
{
    Timing timing("Compiling");
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::neuraylib::ICompiled_material* compiled_material =
        material_instance2->create_compiled_material(flags, context);
    check_success(print_messages(context));

    return compiled_material;
}

// If value is a texture, add all its u/v pairs and frame numbers to frame_uvtiles.
mi::Size build_canvases(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IValue* value,
    Frame_uvtiles& frame_uvtiles)
{
    if( value->get_kind() !=  mi::neuraylib::IValue::VK_TEXTURE)
        return 0;

    mi::base::Handle<const mi::neuraylib::IValue_texture> value_texture(
        value->get_interface<mi::neuraylib::IValue_texture>());
    const char* texture_name = value_texture->get_value();
    if( !texture_name)
        return 0;

    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>( texture_name));
    if( !texture)
        return 0;

    const char* image_name = texture->get_image();
    if( !image_name)
        return 0;

    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>( image_name));
    if( !image)
        return 0;

    bool is_animated = image->is_animated();
    bool is_uvtile = image->is_uvtile();
    if (!is_animated && !is_uvtile)
    {
        // Plain (non-uvtile, non-animated) textures are baked as a single tile.
        return 0;
    }

    mi::Size count = 0;
    mi::Size length = image->get_length();
    for( mi::Size i = 0; i < length; ++i) {

        mi::Size frame_number = image->get_frame_number( i);
        mi::Size frame_length = image->get_frame_length( i);
        count += frame_length;

        for( mi::Size j = 0; j < frame_length; ++j) {
            mi::Sint32 u = 0;
            mi::Sint32 v = 0;
            mi::Sint32 rtn = image->get_uvtile_uv( i, j, u, v);
            assert(0 == rtn);
            if (0 == rtn)
            {
                frame_uvtiles[frame_number].insert(Uvtile(u, v));
            }
            else
            {
                std::cerr << "ERROR: uvtile_id is out of range." << std::endl;
            }
        }
    }

    return count;
}

// Adds u/v pairs and frame numbers of all found textures to frame_uvtiles.
// Note: this traversal visits each temporary and parameter once per reference.
mi::Size build_canvases(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::IExpression* expression,
    const mi::neuraylib::ICompiled_material* cm,
    Frame_uvtiles& frame_uvtiles,
    std::vector<bool>& visited_temps)
{
    switch( expression->get_kind()) {
        case mi::neuraylib::IExpression::EK_CONSTANT: {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
                expression->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<const mi::neuraylib::IValue> value( constant->get_value());
            return build_canvases( transaction, value.get(), frame_uvtiles);
        }
        case mi::neuraylib::IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> direct_call(
                expression->get_interface<mi::neuraylib::IExpression_direct_call>());
            mi::base::Handle<const mi::neuraylib::IExpression_list> args(
                direct_call->get_arguments());
            mi::Size count = 0;
            for( mi::Size i = 0; i < args->get_size(); ++i) {
                mi::base::Handle<const mi::neuraylib::IExpression> arg( args->get_expression( i));
                count += build_canvases( transaction, arg.get(), cm, frame_uvtiles, visited_temps);
            }
            return count;
        }
        case mi::neuraylib::IExpression::EK_TEMPORARY: {
            mi::base::Handle<const mi::neuraylib::IExpression_temporary> temporary_ref(
                expression->get_interface<mi::neuraylib::IExpression_temporary>());
            mi::Size index = temporary_ref->get_index();
            // visit every temporary expression only once
            if( visited_temps[index])
                return 0;
            visited_temps[index] = true;
            mi::base::Handle<const mi::neuraylib::IExpression> temporary(
                cm->get_temporary( index));
            return build_canvases( transaction, temporary.get(), cm, frame_uvtiles, visited_temps);
        }
        case mi::neuraylib::IExpression::EK_PARAMETER: {
            mi::base::Handle<const mi::neuraylib::IExpression_parameter> parameter_ref(
                expression->get_interface<mi::neuraylib::IExpression_parameter>());
            mi::Size index = parameter_ref->get_index();
            mi::base::Handle<const mi::neuraylib::IValue> parameter(
                cm->get_argument( index));
            return build_canvases( transaction, parameter.get(), frame_uvtiles);
        }
        case mi::neuraylib::IExpression::EK_CALL:
            break;
    }

    assert( false);
    return 0;
}

// Remap normal values from the internal MDL range [-1, 1] to the standard
// tangent-space normal map range [0, 1] via v' = (v + 1) * 0.5.
// Returns true on success, false if the canvas is null or not Float32<3>.
bool remap_normal(mi::neuraylib::ICanvas* canvas)
{
    if (!canvas)
        return false;
    if (strcmp(canvas->get_type(), "Float32<3>") != 0) {
        std::cerr << "remap_normal: expected Float32<3> canvas, got '"
                  << canvas->get_type() << "' — skipping remapping.\n";
        return false;
    }
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
    if (!tile)
        return false;
    mi::Float32* data = static_cast<mi::Float32*>(tile->get_data());
    const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y() * 3;
    for (mi::Uint32 i = 0; i < n; ++i)
        data[i] = (data[i] + 1.f) * 0.5f;
    return true;
}

// Scale bool (Sint8) canvas values from the baker's raw range {0, 1} to the
// display range {0, 255} so that true renders as white and false as black.
// Returns true on success, false if the canvas is null or not Sint8.
bool scale_bool_canvas(mi::neuraylib::ICanvas* canvas)
{
    if (!canvas)
        return false;
    if (strcmp(canvas->get_type(), "Sint8") != 0)
        return false;
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
    if (!tile)
        return false;
    // Treat the storage as raw bytes to avoid signed-char overflow.
    // The baker writes 0 (false) or 1 (true); we scale to 0 / 255.
    unsigned char* data = static_cast<unsigned char*>(tile->get_data());
    const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y();
    for (mi::Uint32 i = 0; i < n; ++i)
        data[i] = data[i] ? 255u : 0u;
    return true;
}

// Returns true when a bake path represents a normal (last dot-separated component is "normal").
// Matches geometry.normal and any --expr path ending in .normal.
bool is_normal_path(const std::string& path)
{
    const size_t dot = path.rfind('.');
    const std::string last = (dot == std::string::npos) ? path : path.substr(dot + 1);
    return last == "normal";
}

// Print a baked constant value to stdout.
// The type is obtained from value->get_type_name(), matching the type name strings
// documented at the IBaker interface (Rgb_fp, Float32, Float32<2/3/4>, Boolean).
void print_constant(mi::IData* value)
{
    const char* type_name = value->get_type_name();
    std::cout << "  constant ";
    if (strcmp(type_name, "Rgb_fp") == 0) {
        mi::base::Handle<mi::IColor> c(value->get_interface<mi::IColor>());
        mi::Color v; c->get_value(v);
        std::cout << "color (" << v.r << ", " << v.g << ", " << v.b << ")\n";
    } else if (strcmp(type_name, "Color") == 0) {
        mi::base::Handle<mi::IColor> c(value->get_interface<mi::IColor>());
        mi::Color v; c->get_value(v);
        std::cout << "color (" << v.r << ", " << v.g << ", " << v.b << ", " << v.a << ")\n";
    } else if (strcmp(type_name, "Float32") == 0) {
        mi::base::Handle<mi::IFloat32> fv(value->get_interface<mi::IFloat32>());
        mi::Float32 v = 0.f; fv->get_value(v);
        std::cout << "float " << v << "\n";
    } else if (strcmp(type_name, "Float32<2>") == 0) {
        mi::base::Handle<mi::IFloat32_2> fv(value->get_interface<mi::IFloat32_2>());
        mi::Float32_2 v; fv->get_value(v);
        std::cout << "float2 (" << v.x << ", " << v.y << ")\n";
    } else if (strcmp(type_name, "Float32<3>") == 0) {
        mi::base::Handle<mi::IFloat32_3> fv(value->get_interface<mi::IFloat32_3>());
        mi::Float32_3 v; fv->get_value(v);
        std::cout << "float3 (" << v.x << ", " << v.y << ", " << v.z << ")\n";
    } else if (strcmp(type_name, "Float32<4>") == 0) {
        mi::base::Handle<mi::IFloat32_4> fv(value->get_interface<mi::IFloat32_4>());
        mi::Float32_4 v; fv->get_value(v);
        std::cout << "float4 (" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")\n";
    } else if (strcmp(type_name, "Boolean") == 0) {
        mi::base::Handle<mi::IBoolean> bv(value->get_interface<mi::IBoolean>());
        bool v = false; bv->get_value(v);
        std::cout << "bool " << (v ? "true" : "false") << "\n";
    } else if (strcmp(type_name, "Sint32") == 0) {
        mi::base::Handle<mi::ISint32> iv(value->get_interface<mi::ISint32>());
        mi::Sint32 v = 0; iv->get_value(v);
        std::cout << "int " << v << "\n";
    } else {
        std::cout << "(" << type_name << ")\n";
    }
}

// For each expression: create a baker, detect frames/uvtiles, bake each tile,
// then either export the resulting texture or print the constant value.
void bake_expressions(
    const std::vector<Bake_expression>& expressions,
    mi::neuraylib::Baker_resource baker_resource,
    mi::Uint32 baking_samples,
    mi::Uint32 baking_resolution,
    mi::Float32 min_u,
    mi::Float32 max_u,
    mi::Float32 min_v,
    mi::Float32 max_v,
    bool uv_range_set,
    bool constant_detection,
    bool save_baked_textures,
    const std::string& output_path,
    const std::string& material_name,
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* cm,
    mi::neuraylib::IMdl_distiller_api* distiller_api,
    mi::neuraylib::IImage_api* image_api,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api)
{
    Timing timing("Baking");

    // Constant detection can give false positives for AOV compiled materials
    // (declarative structs that are not SID_MATERIAL): the baker's internal state
    // setup for AOV types may evaluate all sample points identically, making a
    // genuinely varying expression appear constant.  Disable it for AOV materials.
    {
        mi::base::Handle<const mi::neuraylib::IExpression> body(cm->get_body());
        mi::base::Handle<const mi::neuraylib::IType> body_type(body->get_type());
        mi::base::Handle<const mi::neuraylib::IType_struct> body_struct(
            body_type->get_interface<mi::neuraylib::IType_struct>());
        if (!body_struct ||
            body_struct->get_predefined_id() != mi::neuraylib::IType_struct::SID_MATERIAL)
            constant_detection = false;
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << "Material: " << material_name << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& expression : expressions)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> sub_expr(
            cm->lookup_sub_expression(expression.path.c_str()));
        if (!sub_expr) {
            // Skipping: path not present in this material
            continue;
        }

        // Create baker for this expression path.
        mi::base::Handle<const mi::neuraylib::IBaker> baker(
            distiller_api->create_baker(cm, expression.path.c_str(), baker_resource));
        if (!baker) {
            std::cout << "Path: '" << expression.path << "'\n";
            std::cout << "  Skipping: baker creation failed (type unsupported)\n";
            std::cout << std::string(80, '-') << "\n";
            continue;
        }

        const std::string pixel_type = baker->get_pixel_type();
        std::cout << "Path: '" << expression.path << "' [" << pixel_type << "]\n";

        if (baker->is_uniform())
        {
            // ---- Constant ----
            // IBaker::get_type_name() returns a neuray type name (e.g. "Boolean", "Float32<3>")
            // that ITransaction::create() accepts directly — no type switch needed.
            mi::base::Handle<mi::IData> value(
                transaction->create<mi::IData>(baker->get_type_name()));
            if (!value) {
                std::cout << "  Skipping: unsupported pixel type '" << pixel_type << "'\n";
                std::cout << std::string(80, '-') << "\n";
                continue;
            }
            mi::Sint32 result = baker->bake_constant(value.get());
            if (result != 0) {
                std::cout << "  Warning: bake_constant failed (result=" << result << ")\n";
                std::cout << std::string(80, '-') << "\n";
                continue;
            }
            print_constant(value.get());
        }
        else
        {
            // ---- Texture ----

            // Detect uvtile / animation structure from the expression sub-tree.
            // build_canvases() scans for IValue_texture references and fills
            // frame_uvtiles with the frame -> uvtile map it finds.
            // Returns 0 for plain (non-uvtile, non-animated) expressions.
            Frame_uvtiles frame_uvtiles;
            {
                std::vector<bool> visited(cm->get_temporary_count(), false);
                mi::base::Handle<const mi::neuraylib::IExpression> expr(
                    cm->lookup_sub_expression(expression.path.c_str()));
                build_canvases(transaction, expr.get(), cm, frame_uvtiles, visited);
            }

            // For plain expressions synthesise a single frame-0 / tile-(0,0) entry
            // so the loops below are always uniform.
            bool has_uvtiles = !frame_uvtiles.empty();
            if (!has_uvtiles)
                frame_uvtiles[0].insert(Uvtile(0, 0));

            // When the user explicitly specified --uv_range on a UDIM texture, suppress
            // both tile and frame enumeration: produce exactly one bake using the requested
            // range (no _frame or _u_v suffix). The result may be black if no tile covers
            // the requested range.
            if (uv_range_set && has_uvtiles) {
                frame_uvtiles.clear();
                frame_uvtiles[0].insert(Uvtile(0, 0));
                has_uvtiles = false;
            }

            bool multiple_frames = frame_uvtiles.size() > 1;

            for (auto& [frame_number, uvtiles] : frame_uvtiles)
            {
                bool multiple_uvtiles = uvtiles.size() > 1;

                for (const Uvtile& uvtile : uvtiles)
                {
                    // Determine the UV range for this tile.
                    mi::Float32 tile_min_u, tile_max_u, tile_min_v, tile_max_v;
                    if (has_uvtiles) {
                        tile_min_u = mi::Float32(uvtile.first);
                        tile_max_u = tile_min_u + 1.0f;
                        tile_min_v = mi::Float32(uvtile.second);
                        tile_max_v = tile_min_v + 1.0f;
                    } else {
                        tile_min_u = min_u; tile_max_u = max_u;
                        tile_min_v = min_v; tile_max_v = max_v;
                    }
                    mi::Float32 animation_time = mi::Float32(frame_number);

                    // Create canvas.
                    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                        image_api->create_canvas(pixel_type.c_str(),
                            baking_resolution, baking_resolution));
                    // The baker writes linear values; tag the canvas as linear so the
                    // PNG exporter does not apply any gamma encoding on export.
                    canvas->set_gamma(1.0f);

                    // Bake. For the single-tile case use constant-detection if enabled.
                    bool is_constant = false;
                    mi::base::Handle<mi::IData> detected_value;
                    mi::Sint32 result;

                    if (!has_uvtiles && constant_detection) {
                        detected_value = transaction->create<mi::IData>(baker->get_type_name());
                        result = baker->bake_texture_with_constant_detection(
                            canvas.get(), detected_value.get(), is_constant,
                            tile_min_u, tile_max_u, tile_min_v, tile_max_v,
                            animation_time, baking_samples);
                    } else {
                        result = baker->bake_texture(
                            canvas.get(),
                            tile_min_u, tile_max_u, tile_min_v, tile_max_v,
                            animation_time, baking_samples);
                    }

                    if (result != 0) {
                        std::cout << "  Warning: bake_texture failed (result=" << result << ")\n";
                        continue;
                    }

                    // If constant detection fired, just print the value.
                    if (is_constant && detected_value) {
                        print_constant(detected_value.get());
                        continue;
                    }

                    // Remap normal maps from MDL internal range [-1,1] to [0,1].
                    if (is_normal_path(expression.path) && !remap_normal(canvas.get()))
                        std::cerr << "Warning: normal remapping failed for '" << expression.path << "'.\n";

                    // Scale bool (Sint8) values from baker-raw {0,1} to display range {0,255}.
                    scale_bool_canvas(canvas.get());

                    // Export texture.
                    // Filenames follow the UVTILE0 convention: _u<u>_v<v>
                    if (save_baked_textures) {
                        std::stringstream filename;
                        if (!output_path.empty())
                            filename << output_path << "/";
                        filename << material_name << "-" << sanitize_for_filename(expression.path);
                        if (multiple_frames)
                            filename << "_frame" << frame_number;
                        if (multiple_uvtiles)
                            filename << "_u" << uvtile.first
                                     << "_v" << uvtile.second;
                        filename << canvas_extension(pixel_type);
                        check_success(
                            mdl_impexp_api->export_canvas(filename.str().c_str(), canvas.get()) == 0);
                        std::cout << "  -> " << filename.str() << "\n";
                    } else {
                        std::cout << "  -> <not saved>\n";
                    }
                }
            }
        }

        std::cout << std::string(80, '-') << "\n";
    }
}

// Prints program usage
void usage(const char *name)
{
    std::cout
        << "Usage: " << name << " [options] <material_name>\n"
        << "-h                           print this text\n"
        << "-e|--expr <path>             expression path to bake, e.g. surface.scattering.tint;\n"
        << "                             may be given multiple times. If omitted, a curated\n"
        << "                             default set is baked (see g_default_expressions in source).\n"
        << "-b|--baker_resource <device> baking device: gpu|cpu|gpu_with_cpu_fallback (default: cpu)\n"
        << "-s|--samples <num>           baking samples per pixel (default: 4)\n"
        << "-r|--resolution <num>        baking resolution (default: 1024)\n"
        << "--uv_range <min_u max_u min_v max_v>  baking UV range (default: 0 1 0 1)\n"
        << "-a|--auto                    discover all bakeable paths from the compiled material automatically\n"
        << "--do_not_save_textures       if set, avoid saving baked textures to file\n"
        << "--no_constant_detection      if set, do not perform constant detection optimization when baking textures\n"
        << "--dump_material              print the compiled material (hashes, arguments, temporaries, body)\n"
        << "-o|--output_folder <path>    folder to write baked textures to (default: current directory)\n"
        << "-p|--mdl_path <path>         mdl search path, can occur multiple times.\n";

    exit(EXIT_FAILURE);
}

int MAIN_UTF8(int argc, char* argv[])
{
    mi::neuraylib::Baker_resource   baker_resource = mi::neuraylib::BAKE_ON_CPU;
    bool                            baker_resource_set = false;
    mi::Uint32                      baking_samples = 4;
    mi::Uint32                      baking_resolution = 1024;
    bool                            uv_range_set(false);
    mi::Float32                     min_u = 0;
    mi::Float32                     max_u = 1;
    mi::Float32                     min_v = 0;
    mi::Float32                     max_v = 1;
    std::string                     material_name;
    std::vector<Bake_expression>    expression_args;
    bool                            auto_paths = false;
    bool                            save_baked_textures = true;
    std::string                     output_path;      // empty = current directory
    // By default detect whether a baked texture is constant and store it as a value instead.
    bool                            constant_detection = true;
    bool                            dump_material = false;

    mi::examples::mdl::Configure_options configure_options;

    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "-p") == 0 || strcmp(opt, "--mdl_path") == 0) {
                if (i < argc - 1)
                    configure_options.additional_mdl_paths.push_back(argv[++i]);
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else if (strcmp(opt, "-e") == 0 || strcmp(opt, "--expr") == 0) {
                if (i < argc - 1) {
                    Bake_expression expression;
                    expression.path = argv[++i];
                    expression_args.push_back(std::move(expression));
                }
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else if (strcmp(opt, "-b") == 0 || strcmp(opt, "--baker_resource") == 0) {
                if (i < argc - 1) {
                    std::string res = argv[++i];
                    if (res == "gpu")
                        baker_resource = mi::neuraylib::BAKE_ON_GPU;
                    else if (res == "gpu_with_cpu_fallback")
                        baker_resource = mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK;
                    else if (res != "cpu") {
                        std::cerr << "Error: invalid value '" << res
                            << "' for '--baker_resource' (expected: cpu|gpu|gpu_with_cpu_fallback)\n";
                        usage(argv[0]);
                    }
                    baker_resource_set = true;
                }
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else if (strcmp(opt, "-s") == 0 || strcmp(opt, "--samples") == 0) {
                if (i < argc - 1)
                {
                    int val(atoi(argv[++i]));
                    if (val > 0)
                        baking_samples = val;
                    else
                        std::cerr << "Warning: invalid value for '--samples', ignored\n";
                }
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else if (strcmp(opt, "-r") == 0 || strcmp(opt, "--resolution") == 0) {
                if (i < argc - 1)
                {
                    int val(atoi(argv[++i]));
                    if (val > 0)
                        baking_resolution = val;
                    else
                        std::cerr << "Warning: invalid value for '--resolution', ignored\n";
                }
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else if (strcmp(opt, "--uv_range") == 0)
            {
                mi::Float32 uv_range[4];
                for (int idx = 0; idx < 4; idx++)
                {
                    if (i < argc - 1)
                    {
                        mi::Float32 val;
                        int ok = sscanf(argv[++i], "%f", &val);
                        if (ok != 1)
                        {
                            std::cerr << "Error: invalid value '" << argv[i]
                                << "' for '--uv_range'\n";
                            usage(argv[0]);
                        }
                        uv_range[idx] = val;
                    }
                    else
                    {
                        std::cerr << "Error: '--uv_range' requires 4 float values, "
                            << idx << " given\n";
                        usage(argv[0]);
                    }
                }
                min_u = uv_range[0];
                max_u = uv_range[1];
                min_v = uv_range[2];
                max_v = uv_range[3];
                uv_range_set = true;
            }
            else if (strcmp(opt, "-a") == 0 || strcmp(opt, "--auto") == 0) {
                auto_paths = true;
            }
            else if (strcmp(opt, "--do_not_save_textures") == 0) {
                save_baked_textures = false;
            }
            else if (strcmp(opt, "--no_constant_detection") == 0) {
                constant_detection = false;
            }
            else if (strcmp(opt, "--dump_material") == 0) {
                dump_material = true;
            }
            else if (strcmp(opt, "-o") == 0 || strcmp(opt, "--output_folder") == 0) {
                if (i < argc - 1)
                    output_path = argv[++i];
                else {
                    std::cerr << "Error: missing value for '" << opt << "'\n";
                    usage(argv[0]);
                }
            }
            else {
                std::cerr << "Error: unknown argument '" << opt << "'\n";
                usage(argv[0]);
            }
        }
        else
        {
            if (!material_name.empty())
            {
                std::cerr << "Only a single material name is supported.\n";
                usage(argv[0]);
            }
            material_name = opt;
        }
    }

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    {
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        // Get MDL factory
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        {   // inner scope: all DB-object handles must be released before commit()

            if (material_name.empty())
            {
                material_name = "::nvidia::sdk_examples::tutorials_distilling::example_distilling1";
            }

            // split module and material name
            std::string module_qualified_name, material_simple_name;
            if (!mi::examples::mdl::parse_cmd_argument_material_name(
                material_name, module_qualified_name, material_simple_name, true))
                exit_failure();

            // Create an execution context
            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Load mdl module and create a material instance
            mi::base::Handle<mi::neuraylib::IFunction_call> instance(
                create_material_instance(
                    mdl_factory.get(),
                    transaction.get(),
                    mdl_impexp_api.get(),
                    context.get(),
                    module_qualified_name,
                    material_simple_name));

            // Compile the material instance
            mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                compile_material_instance(
                    mdl_factory.get(),
                    transaction.get(),
                    instance.get(),
                    context.get(),
                    false));

            // Dump the compiled material structure (hashes, arguments, temporaries, body).
            if (dump_material) {
                print_header(" Compiled material ");
                mi::examples::mdl::dump_compiled_material(
                    transaction.get(), mdl_factory.get(), compiled_material.get(), std::cout);
            }

            // Acquire distilling API used for baking (no distillation is performed).
            mi::base::Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
                neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

            // Acquire image API needed to create a canvas for baking.
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            // Pick the bake-path set to use:
            //   --expr given  → user-supplied paths
            //   --auto given  → all bakeable paths discovered from the compiled material
            //   (neither)     → curated default list (g_default_expressions)
            std::vector<Bake_expression> dynamic_expressions;
            if (auto_paths) {
                for (const auto& p : collect_bake_paths(compiled_material.get())) {
                    Bake_expression expression;
                    expression.path = p;
                    dynamic_expressions.push_back(std::move(expression));
                }
            }
            const std::vector<Bake_expression>& expressions =
                !expression_args.empty() ? expression_args
                : auto_paths             ? dynamic_expressions
                                         : g_default_expressions;

            print_header(" Expressions to bake ");
            {
                int idx = 1;
                for (const auto& expression : expressions) {
                    std::cout << "  " << std::setw(3) << idx++ << ". " << expression.path << "\n";
                }
            }
            print_header("-");

            // Print which baking device is being used.
            switch (baker_resource) {
                case mi::neuraylib::BAKE_ON_GPU:
                    print_header(" Bake on GPU ");
                    break;
                case mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK:
                    print_header(" Bake on GPU with CPU fallback ");
                    break;
                default:
                    print_header(baker_resource_set ? " Bake on CPU " : " Bake on CPU (default) ");
                    break;
            }

            // Bake every expression: detect frames/uvtiles, bake each tile,
            // then export the texture or print the constant.
            bake_expressions(
                expressions,
                baker_resource,
                baking_samples,
                baking_resolution,
                min_u,
                max_u,
                min_v,
                max_v,
                uv_range_set,
                constant_detection,
                save_baked_textures,
                output_path,
                material_simple_name,
                transaction.get(),
                compiled_material.get(),
                distiller_api.get(),
                image_api.get(),
                mdl_impexp_api.get());

        }   // end inner scope: all DB-object handles released before commit()

        transaction->commit();
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