/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <mi/mdl/mdl_distiller_node_types.h>
#include <mi/mdl/mdl_types.h>

#include "mdl/compiler/compilercore/compilercore_assert.h"

namespace mi {
namespace mdl {

// Global reference to Node_types object, set by init();
Node_types s_node_types;

// Return node type
std::string Node_type::get_return_type() const {
    size_t i = Node_types::static_idx_from_type( type_name.c_str());
    if ( i == material)
        return "material";
    if ( i == material_surface)
        return "material_surface";
    if ( i == material_emission)
        return "material_emission";
    if ( i == material_volume)
        return "material_volume";
    if ( i == material_geometry)
        return "material_geometry";
    if ((i >= bsdf) && (i < last_bsdf))
        return "bsdf";
    if ( (i >= edf) && (i < last_edf))
        return "edf";
    if ( (i >= vdf) && (i < last_vdf))
        return "vdf";
    if ( (i >= hair_bsdf) && (i < last_hair_bsdf))
        return "hair_bsdf";
    if ( i == local_normal)
        return "color";
    return "UNKNOWN";
}

std::string Node_type::get_signature() const {
    std::string s = mdl_type_name + "(";
    size_t n = parameters.size();
    for ( size_t i = 0; i < n; ++i) {
        // swap tint color to third arg position if this is a glossy BRDF
        s += parameters[i].param_type;
        if ( i+1 < n)
            s += ",";
    }
    s += ")";
    return s;
}

// static members, initialized to 0, will be set later by init();
Node_types::Node_type_vector*  Node_types::s_node_types = 0;
Node_types::Map_type_to_idx*   Node_types::s_map_type_to_idx = 0;


/// Add zero-param node-type.
void Node_types::push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                       mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param) {
    MDL_ASSERT(size_t(type) == s_node_types->size());
    (*s_map_type_to_idx)[type_name] = type;
    s_node_types->push_back( Node_type( type_name, mdl_type_name, sem, sel, min_param));
}

/// Add one-param node-type.
void Node_types::push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                       mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                       const Node_param& param1) {
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
}

/// Add two-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
}

/// Add three-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
}

/// Add four-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
}

/// Add five-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
}

/// Add six-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5, const Node_param& param6)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
    parameters.push_back( param6);
}

/// Add seven-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5, const Node_param& param6,
    const Node_param& param7)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
    parameters.push_back( param6);
    parameters.push_back( param7);
}

/// Add eight-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5, const Node_param& param6,
    const Node_param& param7, const Node_param& param8)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
    parameters.push_back( param6);
    parameters.push_back( param7);
    parameters.push_back( param8);
}

/// Add nine-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5, const Node_param& param6,
    const Node_param& param7, const Node_param& param8, const Node_param& param9)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
    parameters.push_back( param6);
    parameters.push_back( param7);
    parameters.push_back( param8);
    parameters.push_back( param9);
}

/// Add nine-param node-type.
void Node_types::push(
    Mdl_node_type type, const char* type_name, const char* mdl_type_name,
    mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
    const Node_param& param1, const Node_param& param2, const Node_param& param3,
    const Node_param& param4, const Node_param& param5, const Node_param& param6,
    const Node_param& param7, const Node_param& param8, const Node_param& param9,
    const Node_param& param10)
{
    push( type, type_name, mdl_type_name, sem, sel, min_param);
    std::vector<Node_param>& parameters = s_node_types->back().parameters;
    parameters.push_back( param1);
    parameters.push_back( param2);
    parameters.push_back( param3);
    parameters.push_back( param4);
    parameters.push_back( param5);
    parameters.push_back( param6);
    parameters.push_back( param7);
    parameters.push_back( param8);
    parameters.push_back( param9);
    parameters.push_back( param10);
}

void Node_types::init() {
    MDL_ASSERT(s_node_types == 0);
    MDL_ASSERT(s_map_type_to_idx == 0);
    s_node_types = new Node_type_vector;
    s_map_type_to_idx = new Map_type_to_idx;
    push( material, "material", "material",
          mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_STRUCT_MATERIAL", 6,
          Node_param(              "bool", "thin_walled",    "false"),
          Node_param(  "material_surface", "surface",        "material_surface()"),
          Node_param(  "material_surface", "backface",       "material_surface()"),
          Node_param(             "color", "ior",            "color(1.0)"),
          Node_param(   "material_volume", "volume",         "material_volume()"),
          Node_param( "material_geometry", "geometry",       "material_geometry()"),
          Node_param(         "hair_bsdf", "hair",           "hair_bsdf()"));
    push( material_surface, "material_surface", "material_surface",
          mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_STRUCT_MATERIAL_SURFACE", 0,
          Node_param(              "bsdf", "scattering", "bsdf()"),
          Node_param( "material_emission", "emission",   "material_emission()"));
    push( material_emission, "material_emission", "material_emission",
          mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_STRUCT_MATERIAL_EMISSION", 0,
          Node_param(               "edf", "emission",   "edf()"),
          Node_param(             "color", "intensity",  "color()"),
          Node_param(    "intensity_mode", "mode",       "intensity_radiant_exitance"));
    push( material_volume, "material_volume", "material_volume",
          mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_STRUCT_MATERIAL_VOLUME", 0,
          Node_param(               "vdf", "scattering",     "vdf()"),
          Node_param(             "color", "absorption_coefficient", "color()"),
          Node_param(             "color", "scattering_coefficient", "color()"),
          Node_param(             "color", "emission_intensity", "color()"));
    push( material_geometry, "material_geometry", "material_geometry",
          mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_STRUCT_MATERIAL_GEOMETRY", 0,
          Node_param(            "float3", "displacement",   "float3(0.0)"),
          Node_param(             "float", "cutout_opacity", "1.0"),
          Node_param(            "float3", "normal",         "::state::normal()"));

    push( bsdf, "bsdf", "bsdf", mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_DEFAULT_BSDF", 0);
    push( diffuse_reflection_bsdf, "diffuse_reflection_bsdf", "::df::diffuse_reflection_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF", 0,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param( "float", "roughness", "0.0"),
          Node_param( "string", "handle", ""));
    push( dusty_diffuse_reflection_bsdf, "dusty_diffuse_reflection_bsdf", "::df::dusty_diffuse_reflection_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF", 0,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param( "string", "handle", ""));
    push( diffuse_transmission_bsdf, "diffuse_transmission_bsdf", "::df::diffuse_transmission_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF", 0,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param( "string", "handle", ""));
    push( specular_bsdf, "specular_bsdf", "::df::specular_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF", 0,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( simple_glossy_bsdf, "simple_glossy_bsdf", "::df::simple_glossy_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( backscattering_glossy_reflection_bsdf, "backscattering_glossy_reflection_bsdf",
          "::df::backscattering_glossy_reflection_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "string", "handle", ""));
    push( sheen_bsdf, "sheen_bsdf",
          "::df::sheen_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF", 3,
          Node_param(  "float", "roughness", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param(  "bsdf", "multiscatter", "::df::diffuse_reflection_bsdf()"),
          Node_param( "string", "handle", ""));
    push( measured_bsdf, "measured_bsdf", "::df::measured_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF", 1,
          Node_param(  "bsdf_measurement", "measurement", ""),
          Node_param(  "float", "multiplier", "1.0"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));

    push( microfacet_beckmann_smith_bsdf, "microfacet_beckmann_smith_bsdf",
          "::df::microfacet_beckmann_smith_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( microfacet_ggx_smith_bsdf, "microfacet_ggx_smith_bsdf", "::df::microfacet_ggx_smith_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( microfacet_beckmann_vcavities_bsdf, "microfacet_beckmann_vcavities_bsdf",
          "::df::microfacet_beckmann_vcavities_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( microfacet_ggx_vcavities_bsdf, "microfacet_ggx_vcavities_bsdf",
          "::df::microfacet_ggx_vcavities_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "::df::scatter_mode", "mode", "::df::scatter_reflect"),
          Node_param( "string", "handle", ""));
    push( ward_geisler_moroder_bsdf, "ward_geisler_moroder_bsdf",
          "::df::ward_geisler_moroder_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF", 4,
          Node_param(  "float", "roughness_u", "0.0"),
          Node_param(  "float", "roughness_v", "0.0"),
          Node_param(  "color", "tint", "color(1.0)"),
          Node_param(  "color", "multiscatter_tint", "color(0.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "string", "handle", ""));

    push( bsdf_tint, "bsdf_tint", "::df::tint",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT,
          "mi::mdl::DS_DIST_BSDF_TINT", 2,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param(  "bsdf", "base", "bsdf()"));
    push( bsdf_tint_ex, "bsdf_tint_ex", "::df::tint",
        mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT,
        "mi::mdl::DS_DIST_BSDF_TINT2", 3,
        Node_param( "color", "reflection_tint", "color(1.0)"),
        Node_param( "color", "transmission_tint", "color(1.0)"),
        Node_param(  "bsdf", "base", "bsdf()"));
    push( thin_film, "thin_film", "::df::thin_film",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_THIN_FILM,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_THIN_FILM", 3,
          Node_param(  "float", "thickness", ""),
          Node_param(  "color", "ior", ""),
          Node_param(  "bsdf",  "base", ""));
    push( bsdf_directional_factor, "bsdf_directional_factor", "::df::directional_factor",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR,
          "mi::mdl::DS_DIST_BSDF_DIRECTIONAL_FACTOR", 0,
          Node_param(  "color", "normal_tint", "color(1.0)"),
          Node_param(  "color", "grazing_tint", "color(1.0)"),
          Node_param(  "float", "exponent", "5.0"),
          Node_param(  "bsdf", "base", "bsdf()"));
    push( measured_curve_factor, "measured_curve_factor", "::df::measured_curve_factor",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR", 1,
          Node_param(  "color[<N>]", "curve_values", ""),
          Node_param(  "bsdf", "base", "bsdf()"));
    push( measured_factor, "measured_factor", "::df::measured_factor",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR", 1,
          Node_param(  "texture_2d", "values", ""),
          Node_param(  "bsdf", "base", "bsdf()"));
    push( fresnel_factor, "fresnel_factor", "::df::fresnel_factor",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR", 2,
          Node_param(  "color", "ior", ""),
          Node_param(  "color", "extinction_coefficient", ""),
          Node_param(  "bsdf", "base", "bsdf()"));

    push( bsdf_mix_1, "bsdf_mix_1", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"));
    push( bsdf_mix_2, "bsdf_mix_2", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "bsdf", "component_2", "bsdf()"));
    push( bsdf_mix_3, "bsdf_mix_3", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "bsdf", "component_2", "bsdf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "bsdf", "component_3", "bsdf()"));
    push( bsdf_clamped_mix_1, "bsdf_clamped_mix_1", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_CLAMPED_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"));
    push( bsdf_clamped_mix_2, "bsdf_clamped_mix_2", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_CLAMPED_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "bsdf", "component_2", "bsdf()"));
    push( bsdf_clamped_mix_3, "bsdf_clamped_mix_3", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_CLAMPED_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "bsdf", "component_2", "bsdf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "bsdf", "component_3", "bsdf()"));
    push( bsdf_unbounded_mix_1, "bsdf_unbounded_mix_1", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_UNBOUNDED_MIX_1", 2,
          Node_param("float", "weight_1", "0.0"),
          Node_param("bsdf", "component_1", "bsdf()"));
    push( bsdf_unbounded_mix_2, "bsdf_unbounded_mix_2", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_UNBOUNDED_MIX_2", 4,
          Node_param("float", "weight_1", "0.0"),
          Node_param("bsdf", "component_1", "bsdf()"),
          Node_param("float", "weight_2", "0.0"),
          Node_param("bsdf", "component_2", "bsdf()"));
    push( bsdf_unbounded_mix_3, "bsdf_unbounded_mix_3", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_UNBOUNDED_MIX_3", 6,
          Node_param("float", "weight_1", "0.0"),
          Node_param("bsdf", "component_1", "bsdf()"),
          Node_param("float", "weight_2", "0.0"),
          Node_param("bsdf", "component_2", "bsdf()"),
          Node_param("float", "weight_3", "0.0"),
          Node_param("bsdf", "component_3", "bsdf()"));

    push( bsdf_color_mix_1, "bsdf_color_mix_1", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"));
    push( bsdf_color_mix_2, "bsdf_color_mix_2", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"));
    push( bsdf_color_mix_3, "bsdf_color_mix_3", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "bsdf", "component_3", "bsdf()"));
    push( bsdf_color_clamped_mix_1, "bsdf_color_clamped_mix_1", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_CLAMPED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"));
    push( bsdf_color_clamped_mix_2, "bsdf_color_clamped_mix_2", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_CLAMPED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"));
    push( bsdf_color_clamped_mix_3, "bsdf_color_clamped_mix_3", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_CLAMPED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "bsdf", "component_3", "bsdf()"));
    push( bsdf_color_unbounded_mix_1, "bsdf_color_unbounded_mix_1", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"));
    push( bsdf_color_unbounded_mix_2, "bsdf_color_unbounded_mix_2", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"));
    push( bsdf_color_unbounded_mix_3, "bsdf_color_unbounded_mix_3", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "bsdf", "component_1", "bsdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "bsdf", "component_2", "bsdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "bsdf", "component_3", "bsdf()"));

    push( weighted_layer, "weighted_layer", "::df::weighted_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER", 2,
          Node_param(  "float", "weight", "1.0"),
          Node_param(   "bsdf", "layer", "bsdf()"),
          Node_param(   "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( color_weighted_layer, "color_weighted_layer", "::df::color_weighted_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER", 2,
          Node_param(  "color", "weight", "color(1.0)"),
          Node_param(   "bsdf", "layer", "bsdf()"),
          Node_param(   "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( fresnel_layer, "fresnel_layer",  "::df::fresnel_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER", 1,
          Node_param(  "float", "ior", "1.0"),
          Node_param(  "float", "weight", "1.0"),
          Node_param(   "bsdf", "layer", "bsdf()"),
          Node_param(   "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( color_fresnel_layer, "color_fresnel_layer",  "::df::color_fresnel_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER", 1,
          Node_param(  "color", "ior", "color(1.0)"),
          Node_param(  "color", "weight", "color(1.0)"),
          Node_param(   "bsdf", "layer", "bsdf()"),
          Node_param(   "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( custom_curve_layer, "custom_curve_layer", "::df::custom_curve_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER", 1,
          Node_param(  "float", "normal_reflectivity", ""),
          Node_param(  "float", "grazing_reflectivity", "1.0"),
          Node_param(  "float", "exponent", "5.0"),
          Node_param(  "float", "weight", "1.0"),
          Node_param(  "bsdf", "layer", "bsdf()"),
          Node_param(  "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( color_custom_curve_layer, "color_custom_curve_layer",
            "::df::color_custom_curve_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER", 1,
          Node_param(  "color", "normal_reflectivity", ""),
          Node_param(  "color", "grazing_reflectivity", "color(1.0)"),
          Node_param(  "float", "exponent", "5.0"),
          Node_param(  "color", "weight", "color(1.0)"),
          Node_param(  "bsdf", "layer", "bsdf()"),
          Node_param(  "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( bsdf_measured_curve_layer, "measured_curve_layer", "::df::measured_curve_layer",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER", 1,
          Node_param(  "color[<N>]", "curve_values", ""),
          Node_param(  "float", "weight", "1.0"),
          Node_param(  "bsdf", "layer", "bsdf()"),
          Node_param(  "bsdf", "base", "bsdf()"),
          Node_param( "float3", "normal", "::state::normal()"));
    push( color_measured_curve_layer, "color_measured_curve_layer",
            "::df::color_measured_curve_layer",
        mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER,
        "mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER", 1,
        Node_param(  "color[<N>]", "curve_values", ""),
        Node_param(  "color", "weight", "color(1.0)"),
        Node_param(  "bsdf", "layer", "bsdf()"),
        Node_param(  "bsdf", "base", "bsdf()"),
        Node_param( "float3", "normal", "::state::normal()"));

    push( edf, "edf", "edf", mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_DEFAULT_EDF", 0);
    push( diffuse_edf, "diffuse_edf", "::df::diffuse_edf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_DIFFUSE_EDF", 0,
          Node_param( "string", "handle", ""));
    push( spot_edf, "spot_edf", "::df::spot_edf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_SPOT_EDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_SPOT_EDF", 1,
          Node_param( "float", "exponent", ""),
          Node_param( "float", "spread", "3.141592653"),
          Node_param( "bool", "global_distribution", "true"),
          Node_param( "float3x3", "global_frame", "float3x3(1.0)"),
          Node_param( "string", "handle", ""));
    push( measured_edf, "measured_edf", "::df::measured_edf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_EDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_MEASURED_EDF", 1,
          Node_param( "light_profile", "profile", ""),
          Node_param( "float", "multiplier", "1.0"),
          Node_param( "bool", "global_distribution", "true"),
          Node_param( "float3x3", "global_frame", "float3x3(1.0)"),
          Node_param( "float3", "tangent_u", "::state::texture_tangent_u(0)"),
          Node_param( "string", "handle", ""));

    push( edf_tint, "edf_tint", "::df::tint",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT,
          "mi::mdl::DS_DIST_EDF_TINT", 2,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param(  "edf", "base", "edf()"));

    push( edf_directional_factor, "edf_directional_factor", "::df::directional_factor",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR,
          "mi::mdl::DS_DIST_EDF_DIRECTIONAL_FACTOR", 0,
          Node_param(  "color", "normal_tint", "color(1.0)"),
          Node_param(  "color", "grazing_tint", "color(1.0)"),
          Node_param(  "float", "exponent", "5.0"),
          Node_param(  "edf", "base", "edf()"));

    push( edf_mix_1, "edf_mix_1", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_mix_2, "edf_mix_2", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_mix_3, "edf_mix_3", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "edf", "component_3", "edf()"));
    push( edf_clamped_mix_1, "edf_clamped_mix_1", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_CLAMPED_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_clamped_mix_2, "edf_clamped_mix_2", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_CLAMPED_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_clamped_mix_3, "edf_clamped_mix_3", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_CLAMPED_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "edf", "component_3", "edf()"));
    push( edf_unbounded_mix_1, "edf_unbounded_mix_1", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_UNBOUNDED_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_unbounded_mix_2, "edf_unbounded_mix_2", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_UNBOUNDED_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_unbounded_mix_3, "edf_unbounded_mix_3", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_UNBOUNDED_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "edf", "component_3", "edf()"));

    push( edf_color_mix_1, "edf_color_mix_1", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_color_mix_2, "edf_color_mix_2", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_color_mix_3, "edf_color_mix_3", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "edf", "component_3", "edf()"));
    push( edf_color_clamped_mix_1, "edf_color_clamped_mix_1", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_CLAMPED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_color_clamped_mix_2, "edf_color_clamped_mix_2", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_CLAMPED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_color_clamped_mix_3, "edf_color_clamped_mix_3", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_CLAMPED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "edf", "component_3", "edf()"));
    push( edf_color_unbounded_mix_1, "edf_color_unbounded_mix_1", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_UNBOUNDED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"));
    push( edf_color_unbounded_mix_2, "edf_color_unbounded_mix_2", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_UNBOUNDED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"));
    push( edf_color_unbounded_mix_3, "edf_color_unbounded_mix_3", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_EDF_COLOR_UNBOUNDED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "edf", "component_1", "edf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "edf", "component_2", "edf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "edf", "component_3", "edf()"));

    push( vdf, "vdf", "vdf", mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_DEFAULT_VDF", 0);
    push( vdf_anisotropic, "vdf_anisotropic", "::df::anisotropic_vdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_ANISOTROPIC_VDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_ANISOTROPIC_VDF", 0,
          Node_param( "float", "directional_bias", "0.0"),
          Node_param( "string", "handle", ""));
    push( vdf_fog, "vdf_fog", "::df::fog_vdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_FOG_VDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_FOG_VDF", 0,
          Node_param( "float", "particle_size", "8.0"),
          Node_param( "string", "handle", ""));

    push( vdf_tint, "vdf_tint", "::df::tint",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT,
          "mi::mdl::DS_DIST_VDF_TINT", 2,
          Node_param( "color", "tint", "color(1.0)"),
          Node_param(  "vdf", "base", "vdf()"));

    push( vdf_mix_1, "vdf_mix_1", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_mix_2, "vdf_mix_2", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_mix_3, "vdf_mix_3", "::df::normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "vdf", "component_3", "vdf()"));
    push( vdf_clamped_mix_1, "vdf_clamped_mix_1", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_CLAMPED_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_clamped_mix_2, "vdf_clamped_mix_2", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_CLAMPED_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_clamped_mix_3, "vdf_clamped_mix_3", "::df::clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_CLAMPED_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "vdf", "component_3", "vdf()"));
    push( vdf_unbounded_mix_1, "vdf_unbounded_mix_1", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_UNBOUNDED_MIX_1", 2,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_unbounded_mix_2, "vdf_unbounded_mix_2", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_UNBOUNDED_MIX_2", 4,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_unbounded_mix_3, "vdf_unbounded_mix_3", "::df::unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_UNBOUNDED_MIX_3", 6,
          Node_param( "float", "weight_1", "0.0"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "float", "weight_2", "0.0"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "float", "weight_3", "0.0"),
          Node_param(  "vdf", "component_3", "vdf()"));

    push( vdf_color_mix_1, "vdf_color_mix_1", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_color_mix_2, "vdf_color_mix_2", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_color_mix_3, "vdf_color_mix_3", "::df::color_normalized_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "vdf", "component_3", "vdf()"));
    push( vdf_color_clamped_mix_1, "vdf_color_clamped_mix_1", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_CLAMPED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_color_clamped_mix_2, "vdf_color_clamped_mix_2", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_CLAMPED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_color_clamped_mix_3, "vdf_color_clamped_mix_3", "::df::color_clamped_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_CLAMPED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "vdf", "component_3", "vdf()"));
    push( vdf_color_unbounded_mix_1, "vdf_color_unbounded_mix_1", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_UNBOUNDED_MIX_1", 2,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"));
    push( vdf_color_unbounded_mix_2, "vdf_color_unbounded_mix_2", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_UNBOUNDED_MIX_2", 4,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"));
    push( vdf_color_unbounded_mix_3, "vdf_color_unbounded_mix_3", "::df::color_unbounded_mix",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX,
          "mi::mdl::DS_DIST_VDF_COLOR_UNBOUNDED_MIX_3", 6,
          Node_param( "color", "weight_1", "color(0.0)"),
          Node_param(  "vdf", "component_1", "vdf()"),
          Node_param( "color", "weight_2", "color(0.0)"),
          Node_param(  "vdf", "component_2", "vdf()"),
          Node_param( "color", "weight_3", "color(0.0)"),
          Node_param(  "vdf", "component_3", "vdf()"));

    push( hair_bsdf, "hair_bsdf", "hair_bsdf", mi::mdl::IDefinition::DS_ELEM_CONSTRUCTOR,
          "mi::mdl::DS_DIST_DEFAULT_HAIR_BSDF", 0);
    push( hair_bsdf_chiang, "chiang_hair_bsdf", "::df::chiang_hair_bsdf",
          mi::mdl::IDefinition::DS_INTRINSIC_DF_CHIANG_HAIR_BSDF,
          "mi::mdl::IDefinition::DS_INTRINSIC_DF_CHIANG_HAIR_BSDF", 0,
          Node_param( "float", "diffuse_reflection_weight", "0.0"),
          Node_param( "color", "diffuse_reflection_tint", "color(1.0)"),
          Node_param( "float2", "roughness_R", "float2(0.0)"),
          Node_param( "float2", "roughness_TT", "float2(0.0)"),
          Node_param( "float2", "roughness_TRT", "float2(0.0)"),
          Node_param( "float", "cuticle_angle", "0.0"),
          Node_param( "color", "absorption_coefficient", "color(0.0)"),
          Node_param( "float", "ior", "1.55"),
          Node_param( "string", "handle", ""));

    push(hair_bsdf_tint, "hair_bsdf_tint", "::df::tint",
        mi::mdl::IDefinition::DS_INTRINSIC_DF_TINT,
        "mi::mdl::DS_DIST_HAIR_BSDF_TINT", 2,
        Node_param("color", "tint", "color(1.0)"),
        Node_param("hair_bsdf", "base", "hair_bsdf()"));


    // helper node types

    push( material_conditional_operator, "material_conditional_operator",
         "operator?(bool,<0>,<0>)",
          mi::mdl::IDefinition::Semantics(mi::mdl::IExpression::OK_TERNARY),
          "mi::mdl::DS_DIST_MATERIAL_CONDITIONAL_OPERATOR", 3,
          Node_param(  "bool", "cond", ""),
          Node_param(  "material", "true_exp", ""),
          Node_param(  "material", "false_exp", ""));

    push( bsdf_conditional_operator, "bsdf_conditional_operator",
          "operator?(bool,<0>,<0>)",
          mi::mdl::IDefinition::Semantics(mi::mdl::IExpression::OK_TERNARY),
          "mi::mdl::DS_DIST_BSDF_CONDITIONAL_OPERATOR", 3,
          Node_param(  "bool", "cond", ""),
          Node_param(  "bsdf", "true_exp", ""),
          Node_param(  "bsdf", "false_exp", ""));

    push( edf_conditional_operator, "edf_conditional_operator",
          "operator?(bool,<0>,<0>)",
          mi::mdl::IDefinition::Semantics(mi::mdl::IExpression::OK_TERNARY),
          "mi::mdl::DS_DIST_EDF_CONDITIONAL_OPERATOR", 3,
          Node_param(  "bool", "cond", ""),
          Node_param(  "edf", "true_exp", ""),
          Node_param(  "edf", "false_exp", ""));

    push( vdf_conditional_operator, "vdf_conditional_operator",
          "operator?(bool,<0>,<0>)",
          mi::mdl::IDefinition::Semantics(mi::mdl::IExpression::OK_TERNARY),
          "mi::mdl::DS_DIST_VDF_CONDITIONAL_OPERATOR", 3,
          Node_param(  "bool", "cond", ""),
          Node_param(  "vdf", "true_exp", ""),
          Node_param(  "vdf", "false_exp", ""));

    push( local_normal, "local_normal", "::nvidia::distilling_support::local_normal",
          mi::mdl::IDefinition::DS_UNKNOWN,
          "mi::mdl::DS_DIST_LOCAL_NORMAL", 2,
          Node_param(  "float", "weight", "0.0"),
          Node_param( "float3", "normal", "::state::normal()"));


}

// Clean up dynamically allocated memory. Can be initialized again.
void Node_types::exit() {
    delete s_node_types;
    delete s_map_type_to_idx;
    s_node_types = 0;
    s_map_type_to_idx = 0;
}

void Node_types::print_all_nodes( std::ostream& out) {
    MDL_ASSERT(s_node_types != 0);
    for ( size_t i = 0; i < s_node_types->size(); ++i) {
        const Node_type& node_type = (*s_node_types)[i];
        out << node_type.type_name << "(";
        size_t n_params = node_type.parameters.size();
        for ( size_t k = 0; k < n_params; ++k) {
            if ( n_params > 1)
                out << "\n    ";
            const Node_param& param = node_type.parameters[k];
            out << param.param_type << ' ' << param.param_name;
            if ( ! param.param_default.empty()) {
                if ( k < node_type.min_parameters) {
                    // make it in the print out clear that this default can't be used in rules
                    out << " ( = " << param.param_default << ")";
                } else {
                    out << " = " << param.param_default;
                }
            }
            if ( k+1 < n_params)
                out << ", ";
        }
        out << ")\n";
    }
}

void Node_types::print_all_mdl_nodes( std::ostream& out) {
    const char* spaces_43 = "                                           ";
    MDL_ASSERT(s_node_types != 0);
    for ( size_t i = 0; i < s_node_types->size(); ++i) {
        const Node_type& node_type = (*s_node_types)[i];
        size_t l = node_type.mdl_type_name.size();
        if (l > 43)
            l = 43;
        out << (spaces_43 + l) << node_type.mdl_type_name << " --> " << node_type.type_name << '\n';
    }
}

int Node_types::idx_from_type( const char* type_name) {
    MDL_ASSERT(s_node_types != 0);
    Map_type_to_idx::const_iterator pos = s_map_type_to_idx->find( type_name);
    if ( pos == s_map_type_to_idx->end())
        return -1;
    return pos->second;
}

int Node_types::static_idx_from_type( const char* type_name) {
    MDL_ASSERT(s_node_types != 0);
    Map_type_to_idx::const_iterator pos = s_map_type_to_idx->find( type_name);
    if ( pos == s_map_type_to_idx->end())
        return -1;
    return pos->second;
}

const Node_type* Node_types::static_type_from_idx( int idx) {
    MDL_ASSERT(s_node_types != 0);
    MDL_ASSERT(idx >= 0);
    if (size_t(idx) >= s_node_types->size())
        return nullptr;
    return &((* s_node_types)[idx]);
}

const Node_type* Node_types::type_from_idx( int idx) const {
    MDL_ASSERT(s_node_types != 0);
    MDL_ASSERT(idx >= 0 && idx < int(s_node_types->size()));
    return &((* s_node_types)[idx]);
}

const Node_type* Node_types::type_from_name( const char* type_name) {
    MDL_ASSERT(s_node_types != 0);
    return static_type_from_idx( static_idx_from_type( type_name));
}

const std::string& Node_types::type_name( int idx) {
    return static_type_from_idx(idx)->type_name;
}

const std::string& Node_types::mdl_type_name( int idx) {
    return static_type_from_idx(idx)->mdl_type_name;
}

bool Node_types::is_type( const char* type_name) {
    MDL_ASSERT(s_node_types != 0);
    Map_type_to_idx::const_iterator pos = s_map_type_to_idx->find( type_name);
    return ( pos != s_map_type_to_idx->end());
}

size_t Node_types::get_n_params( Mdl_node_type node_type) {
    const Node_type* type = static_type_from_idx( node_type);
    return type->parameters.size();
}

const std::string& Node_types::get_param_type( Mdl_node_type node_type, size_t arg_idx) {
    const Node_type* type = static_type_from_idx( node_type);
    MDL_ASSERT(arg_idx < type->parameters.size());
    return type->parameters[arg_idx].param_type;
}

const std::string& Node_types::get_param_name( Mdl_node_type node_type, size_t arg_idx) {
    const Node_type* type = static_type_from_idx( node_type);
    MDL_ASSERT(arg_idx < type->parameters.size());
    return type->parameters[arg_idx].param_name;
}

const std::string& Node_types::get_param_default( Mdl_node_type node_type, size_t arg_idx) {
    const Node_type* type = static_type_from_idx( node_type);
    MDL_ASSERT(arg_idx < type->parameters.size());
    return type->parameters[arg_idx].param_default;
}

bool Node_types::is_param_node_type( const std::string& param_type_name) {
    return (param_type_name == "material" ||
            param_type_name == "material_surface" ||
            param_type_name == "material_emission" ||
            param_type_name == "material_volume" ||
            param_type_name == "material_geometry" ||
            param_type_name == "bsdf" ||
            param_type_name == "edf" ||
            param_type_name == "vdf" ||
            param_type_name == "hair_bsdf");
}

bool Node_types::is_param_node_type( Mdl_node_type node_type, size_t arg_idx) {
    return is_param_node_type( get_param_type( node_type, arg_idx));
}

} // namespace mdl
} // namespace mi

