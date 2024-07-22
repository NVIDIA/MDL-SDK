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
    bsdf,
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
    bsdf_mix_4,
    bsdf_clamped_mix_1,
    bsdf_clamped_mix_2,
    bsdf_clamped_mix_3,
    bsdf_clamped_mix_4,
    bsdf_unbounded_mix_1,
    bsdf_unbounded_mix_2,
    bsdf_unbounded_mix_3,
    bsdf_unbounded_mix_4,
    bsdf_color_mix_1,
    bsdf_color_mix_2,
    bsdf_color_mix_3,
    bsdf_color_mix_4,
    bsdf_color_clamped_mix_1,
    bsdf_color_clamped_mix_2,
    bsdf_color_clamped_mix_3,
    bsdf_color_clamped_mix_4,
    bsdf_color_unbounded_mix_1,
    bsdf_color_unbounded_mix_2,
    bsdf_color_unbounded_mix_3,
    bsdf_color_unbounded_mix_4,
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
    edf,
    diffuse_edf,
    spot_edf,
    measured_edf,
    edf_tint,
    edf_directional_factor,
    edf_mix_1,
    edf_mix_2,
    edf_mix_3,
    edf_mix_4,
    edf_clamped_mix_1,
    edf_clamped_mix_2,
    edf_clamped_mix_3,
    edf_clamped_mix_4,
    edf_unbounded_mix_1,
    edf_unbounded_mix_2,
    edf_unbounded_mix_3,
    edf_unbounded_mix_4,
    edf_color_mix_1,
    edf_color_mix_2,
    edf_color_mix_3,
    edf_color_mix_4,
    edf_color_clamped_mix_1,
    edf_color_clamped_mix_2,
    edf_color_clamped_mix_3,
    edf_color_clamped_mix_4,
    edf_color_unbounded_mix_1,
    edf_color_unbounded_mix_2,
    edf_color_unbounded_mix_3,
    edf_color_unbounded_mix_4,
    last_edf = edf_color_unbounded_mix_4,
    //
    vdf,
    vdf_anisotropic,
    vdf_fog,
    vdf_tint,
    vdf_mix_1,
    vdf_mix_2,
    vdf_mix_3,
    vdf_mix_4,
    vdf_clamped_mix_1,
    vdf_clamped_mix_2,
    vdf_clamped_mix_3,
    vdf_clamped_mix_4,
    vdf_unbounded_mix_1,
    vdf_unbounded_mix_2,
    vdf_unbounded_mix_3,
    vdf_unbounded_mix_4,
    vdf_color_mix_1,
    vdf_color_mix_2,
    vdf_color_mix_3,
    vdf_color_mix_4,
    vdf_color_clamped_mix_1,
    vdf_color_clamped_mix_2,
    vdf_color_clamped_mix_3,
    vdf_color_clamped_mix_4,
    vdf_color_unbounded_mix_1,
    vdf_color_unbounded_mix_2,
    vdf_color_unbounded_mix_3,
    vdf_color_unbounded_mix_4,
    last_vdf = vdf_color_unbounded_mix_4,
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
    char const *param_type;
    char const *param_name;
    char const *param_default;

    Node_param( const char* param_tp, const char* param_nm, const char* param_dflt)
        : param_type(param_tp), param_name( param_nm), param_default( param_dflt) {}
};

/// Node type
struct Node_type {
    /// Type name of the node type
    char const *type_name;

    /// MDL type name of the node type
    char const *mdl_type_name;

    /// Semantics for the node in the DAG back-end
    mi::mdl::IDefinition::Semantics semantics;

    /// Selector for the node in the matcher, equivalent to DAG_call semantics except for
    /// DAG_constant and the material structures
    char const *selector_enum;

    /// Mandatory parameters, all others beyond that have defaults that are
    /// known to the system
    size_t min_parameters;

    /// Vector of parameter definitions
    std::vector<Node_param> parameters;

    Node_type( const char* type, const char* mdl_type,
               mi::mdl::IDefinition::Semantics sem, const char* sel, size_t min_param)
        : type_name(type)
        , mdl_type_name(mdl_type)
        , semantics(sem)
        , selector_enum(sel)
        , min_parameters( min_param) {}

    std::string get_signature() const;
};

/// All known Node-types. Kind of singleton, explicit init() method for setup
/// and exit() method for cleanup.
class Node_types {

public:

    // Constructor.
    Node_types();

    /// Returns the type index for a type name and -1 if not a type.
    virtual int idx_from_type( const char* type_name) const;

    /// Returns the type for an index.
    virtual const Node_type* type_from_idx( int idx) const;

    /// Returns the type for an index.
    virtual const Node_type* type_from_name( const char* type_name);

    /// Returns the type name for the type of the given index.
    virtual char const *type_name( int idx);

    /// Returns the MDL type name for the type of the given index.
    virtual char const *mdl_type_name( int idx);

    virtual bool is_type( const char* type_name);

    virtual size_t get_n_params( Mdl_node_type node_type);

    virtual char const *get_param_type( Mdl_node_type node_type, size_t arg_idx);

    virtual char const *get_param_name( Mdl_node_type node_type, size_t arg_idx);

    virtual char const *get_param_default( Mdl_node_type node_type, size_t arg_idx);

    // returns true if the provided parameter type is a node type
    virtual bool is_param_node_type( char const *param_type_name);

    virtual bool is_param_node_type( Mdl_node_type node_type, size_t arg_idx);

    virtual void print_all_nodes( std::ostream& out);

    virtual void print_all_mdl_nodes( std::ostream& out);

    virtual char const *get_return_type(size_t i) const;
private:

    /// Add node-type, variants from zero parameters to ten parameters
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5, const Node_param &param6);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5, const Node_param &param6,
        const Node_param &param7);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5, const Node_param &param6,
        const Node_param &param7, const Node_param &param8);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5, const Node_param &param6,
        const Node_param &param7, const Node_param &param8, const Node_param &param9);
    void push(Mdl_node_type type, const char *type_name, const char *mdl_type_name,
        mi::mdl::IDefinition::Semantics sem, const char *sel, size_t min_param,
        const Node_param &param1, const Node_param &param2, const Node_param &param3,
        const Node_param &param4, const Node_param &param5, const Node_param &param6,
        const Node_param &param7, const Node_param &param8, const Node_param &param9,
        const Node_param &param10);

    typedef std::map<std::string, int> Map_type_to_idx;
    typedef std::vector<Node_type> Node_type_vector;

    /// Sequence of all registered node types.
    Node_type_vector m_node_types;

    /// Name lookup map that returns the node type index for its given type name.
    Map_type_to_idx m_map_type_to_idx;
};

} // namespace mdl
} // namespace mi

#endif // MDL_DISTILLER_NODE_TYPES_H

