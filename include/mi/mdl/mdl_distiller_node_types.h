/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file mi/mdl/mdl_distiller_node_types.h
/// \brief Node types used during the distilling process

#ifndef MDL_DISTILLER_NODE_TYPES_H
#define MDL_DISTILLER_NODE_TYPES_H


#include <string>
#include <vector>
#include <iostream>
#include <map>

#include <mi/mdl/mdl_definitions.h> // mi::mdl::IDefinition::Semantics

namespace mi {
namespace mdl {

/// Node types in an MDL expression tree.
/// Only material and DF nodes at the moment. A split between simple nodes
/// and the full set of MDL nodes is hard coded at the moment.
typedef int Mdl_node_type;
enum Mdl_node_type_enum {
    // simple set of MDL nodes
    node_null = -1,
    material = 0,
    material_surface,
    material_emission,
    material_volume,
    material_geometry,
    bsdf, // used to be bsdf_null
    diffuse_reflection_bsdf,
    dusty_diffuse_reflection_bsdf,
    diffuse_transmission_bsdf,
    specular_bsdf,
    simple_glossy_bsdf,
    backscattering_glossy_reflection_bsdf,
    sheen_bsdf,
    measured_bsdf,
    //
    microfacet_beckmann_smith_bsdf,
    microfacet_ggx_smith_bsdf,
    microfacet_beckmann_vcavities_bsdf,
    microfacet_ggx_vcavities_bsdf,
    ward_geisler_moroder_bsdf,
    //
    bsdf_tint,
    bsdf_tint_ex,
    thin_film,
    bsdf_directional_factor,
    measured_curve_factor,
    measured_factor,
    fresnel_factor,
    //
    bsdf_mix_1,
    bsdf_mix_2,
    bsdf_mix_3,
    bsdf_clamped_mix_1,
    bsdf_clamped_mix_2,
    bsdf_clamped_mix_3,
    bsdf_unbounded_mix_1,
    bsdf_unbounded_mix_2,
    bsdf_unbounded_mix_3,
    bsdf_color_mix_1,
    bsdf_color_mix_2,
    bsdf_color_mix_3,
    bsdf_color_clamped_mix_1,
    bsdf_color_clamped_mix_2,
    bsdf_color_clamped_mix_3,
    bsdf_color_unbounded_mix_1,
    bsdf_color_unbounded_mix_2,
    bsdf_color_unbounded_mix_3,
    weighted_layer,
    color_weighted_layer,
    fresnel_layer,
    color_fresnel_layer,
    custom_curve_layer,
    color_custom_curve_layer,
    bsdf_measured_curve_layer,
    color_measured_curve_layer,
    last_bsdf = color_measured_curve_layer,
    //
    edf, // used to be edf_null
    diffuse_edf,
    spot_edf,
    measured_edf,
    edf_tint,
    edf_directional_factor,
    edf_mix_1,
    edf_mix_2,
    edf_mix_3,
    edf_clamped_mix_1,
    edf_clamped_mix_2,
    edf_clamped_mix_3,
    edf_unbounded_mix_1,
    edf_unbounded_mix_2,
    edf_unbounded_mix_3,
    edf_color_mix_1,
    edf_color_mix_2,
    edf_color_mix_3,
    edf_color_clamped_mix_1,
    edf_color_clamped_mix_2,
    edf_color_clamped_mix_3,
    edf_color_unbounded_mix_1,
    edf_color_unbounded_mix_2,
    edf_color_unbounded_mix_3,
    last_edf = edf_color_unbounded_mix_3,
    //
    vdf, // used to be vdf_null
    vdf_anisotropic,
    vdf_fog,
    vdf_tint,
    vdf_mix_1,
    vdf_mix_2,
    vdf_mix_3,
    vdf_clamped_mix_1,
    vdf_clamped_mix_2,
    vdf_clamped_mix_3,
    vdf_unbounded_mix_1,
    vdf_unbounded_mix_2,
    vdf_unbounded_mix_3,
    vdf_color_mix_1,
    vdf_color_mix_2,
    vdf_color_mix_3,
    vdf_color_clamped_mix_1,
    vdf_color_clamped_mix_2,
    vdf_color_clamped_mix_3,
    vdf_color_unbounded_mix_1,
    vdf_color_unbounded_mix_2,
    vdf_color_unbounded_mix_3,
    last_vdf = vdf_color_unbounded_mix_3,
    //
    hair_bsdf,
    hair_bsdf_chiang,
    hair_bsdf_tint,
    last_hair_bsdf = hair_bsdf_tint,
    //
    // helper node types
    //
    material_conditional_operator,
    bsdf_conditional_operator,
    edf_conditional_operator,
    vdf_conditional_operator,
    local_normal,
};


/// A node parameter storing a type, name, and default value
struct Node_param {
    std::string param_type;
    std::string param_name;
    std::string param_default;

    Node_param( std::string param_tp, std::string param_nm, std::string param_dflt)
        : param_type(param_tp), param_name( param_nm), param_default( param_dflt) {}
    Node_param( const char* param_tp, const char* param_nm, const char* param_dflt)
        : param_type(param_tp), param_name( param_nm), param_default( param_dflt) {}
};

/// Node type
struct Node_type {
    /// Type name of the node type
    std::string type_name;

    /// MDL type name of the node type
    std::string mdl_type_name;

    /// Semantics for the node in the DAG back-end
    mi::mdl::IDefinition::Semantics semantics;

#ifdef MI_MDLTLC_NODE_TYPES
    /// Selector for the node in the matcher, equivalent to DAG_call semantics except for
    /// DAG_constant and the material structures
    std::string selector_enum;
#endif

    /// Mandatory parameters, all others beyond that have defaults that are
    /// known to the system
    size_t min_parameters;

    /// Vector of parameter definitions
    std::vector<Node_param> parameters;

    Node_type( std::string type, std::string mdl_type,
               mi::mdl::IDefinition::Semantics sem, const std::string& sel, size_t min_param)
        : type_name(type), mdl_type_name(mdl_type), semantics(sem),
#ifdef MI_MDLTLC_NODE_TYPES
          selector_enum(sel),
#endif
          min_parameters( min_param) { (void)sel; }
    Node_type( const char* type, const char* mdl_type,
               mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param)
        : type_name(type), mdl_type_name(mdl_type), semantics(sem),
#ifdef MI_MDLTLC_NODE_TYPES
          selector_enum(sel),
#endif
          min_parameters( min_param) { (void)sel; }

    std::string get_return_type() const;

    std::string get_signature() const;
};

/// All known Node-types. Kind of singleton, explicit init() method for setup
/// and exit() method for cleanup.
class Node_types {
    typedef std::map<std::string,int>    Map_type_to_idx;
    typedef std::vector<Node_type>  Node_type_vector;

    /// Sequence of all registered node types.
    static Node_type_vector*  s_node_types;

    /// Name lookup map that returns the node type index for its given type name.
    static Map_type_to_idx*   s_map_type_to_idx;

    /// Add node-type, variants from zero parameters to ten parameters
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5, const Node_param& param6);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5, const Node_param& param6,
                      const Node_param& param7);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5, const Node_param& param6,
                      const Node_param& param7, const Node_param& param8);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5, const Node_param& param6,
                      const Node_param& param7, const Node_param& param8, const Node_param& param9);
    static void push( Mdl_node_type type, const char* type_name, const char* mdl_type_name,
                      mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param,
                      const Node_param& param1, const Node_param& param2, const Node_param& param3,
                      const Node_param& param4, const Node_param& param5, const Node_param& param6,
                      const Node_param& param7, const Node_param& param8, const Node_param& param9,
                      const Node_param& param10);

public:

    /// Register all known node types with their parameter definitions.
    static void init();

    /// Clean up dynamically allocated memory. Can be initialized again.
    static void exit();

    /// Returns the type index for a type name and -1 if not a type.
    virtual int idx_from_type( const char* type_name);
    static int static_idx_from_type( const char* type_name);

    /// Returns the type for an index.
    virtual const Node_type* type_from_idx( int idx) const;
    static const Node_type* static_type_from_idx( int idx);

    /// Returns the type for an index.
    static const Node_type* type_from_name( const char* type_name);

    /// Returns the type name for the type of the given index.
    static const std::string& type_name( int idx);

    /// Returns the MDL type name for the type of the given index.
    const std::string& mdl_type_name( int idx);

    static bool is_type( const char* type_name);

    static size_t get_n_params( Mdl_node_type node_type);

    static const std::string& get_param_type( Mdl_node_type node_type, size_t arg_idx);

    static const std::string& get_param_name( Mdl_node_type node_type, size_t arg_idx);

    static const std::string& get_param_default( Mdl_node_type node_type, size_t arg_idx);

    // returns true if the provided parameter type is a node type
    static bool is_param_node_type( const std::string& param_type_name);

    static bool is_param_node_type( Mdl_node_type node_type, size_t arg_idx);

    static void print_all_nodes( std::ostream& out);

    static void print_all_mdl_nodes( std::ostream& out);
};

} // namespace mdl
} // namespace mi

#endif // MDL_DISTILLER_NODE_TYPES_H

