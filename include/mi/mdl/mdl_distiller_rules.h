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

/// \file mi/mdl/mdl_distiller_rules.h
/// \brief MDL distiller rule definitions and rule base class

#ifndef MDL_DISTILLER_RULES_H
#define MDL_DISTILLER_RULES_H

#include <cstddef>

#include <mi/mdl/mdl_distiller_options.h>

namespace mi {
namespace mdl {

class IDistiller_plugin_api;
class Debug_output_callback;
class DAG_node;
class Node_types;

/// Rule evaluation strategies.
enum Rule_eval_strategy {
    RULE_EVAL_TOP_DOWN, ///< Evaluate rules top down.
    RULE_EVAL_BOTTOM_UP ///< Evaluate rules bottom up.
};

/// Return codes for the top-down evaluation strategy
enum Rule_result_code {
    RULE_RECURSE,       ///< Default, recurse into the sub-expressions
    RULE_REPEAT_RULES,  ///< Repeat rule matching with current node
    RULE_SKIP_RECURSION ///< Skip recursion into sub-expressions
};

/// Enum that extends mi::mdl::IDefinition::Semantics with semantics for
/// custom extended nodes in the distiller, such as elementary DFs, material
/// structures and mixers. The negative semantics are needed for the 
/// right ordering of the canonical ordering of mixers.
enum Distiller_extended_node_semantics {
    DS_DIST_DEFAULT_BSDF = -1000,                    ///< Default bsdf().
    DS_DIST_DEFAULT_EDF  = -1001,                    ///< Default edf().
    DS_DIST_DEFAULT_VDF  = -1002,                    ///< Default vdf().
    DS_DIST_DEFAULT_HAIR_BSDF  = -1003,              ///< Default hair_bsdf().
    DS_DIST_STRUCT_MATERIAL_EMISSION = 0x10010,      ///< material_emission struct
    DS_DIST_STRUCT_MATERIAL_SURFACE,                 ///< material_surface struct
    DS_DIST_STRUCT_MATERIAL_VOLUME,                  ///< material_volume struct
    DS_DIST_STRUCT_MATERIAL_GEOMETRY,                ///< material_geometry struct
    DS_DIST_STRUCT_MATERIAL,                         ///< material struct
    DS_DIST_BSDF_TINT = 0x10017,                     ///< tint modifier
    DS_DIST_BSDF_TINT2,                              ///< tint modifier with extra reflection
    DS_DIST_EDF_TINT,                                ///< tint modifier
    DS_DIST_VDF_TINT,                                ///< tint modifier
    DS_DIST_HAIR_BSDF_TINT,                          ///< tint modifier
    DS_DIST_BSDF_MIX_1 = 0x10020,                    ///< normalized_mix with one component
    DS_DIST_BSDF_MIX_2,                              ///< normalized_mix with two components
    DS_DIST_BSDF_MIX_3,                              ///< normalized_mix with three components
    DS_DIST_BSDF_CLAMPED_MIX_1,                      ///< clamped_mix with one component
    DS_DIST_BSDF_CLAMPED_MIX_2,                      ///< clamped_mix with two components
    DS_DIST_BSDF_CLAMPED_MIX_3,                      ///< clamped_mix with three components
    DS_DIST_BSDF_UNBOUNDED_MIX_1,                    ///< unbounded_mix with one component
    DS_DIST_BSDF_UNBOUNDED_MIX_2,                    ///< unbounded_mix with two components
    DS_DIST_BSDF_UNBOUNDED_MIX_3,                    ///< unbounded_mix with three components
    DS_DIST_BSDF_COLOR_MIX_1 = 0x10030,              ///< color_normalized_mix with one component
    DS_DIST_BSDF_COLOR_MIX_2,                        ///< color_normalized_mix with two components
    DS_DIST_BSDF_COLOR_MIX_3,                        ///< color_normalized_mix with three components
    DS_DIST_BSDF_COLOR_CLAMPED_MIX_1,                ///< color_clamped_mix with one component
    DS_DIST_BSDF_COLOR_CLAMPED_MIX_2,                ///< color_clamped_mix with two components
    DS_DIST_BSDF_COLOR_CLAMPED_MIX_3,                ///< color_clamped_mix with three components
    DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_1,              ///< color_unbounded_mix with one component
    DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_2,              ///< color_unbounded_mix with two components
    DS_DIST_BSDF_COLOR_UNBOUNDED_MIX_3,              ///< color_unbounded_mix with three components
    DS_DIST_EDF_MIX_1 = 0x10040,                     ///< normalized_mix with one component
    DS_DIST_EDF_MIX_2,                               ///< normalized_mix with two components
    DS_DIST_EDF_MIX_3,                               ///< normalized_mix with three components
    DS_DIST_EDF_CLAMPED_MIX_1,                       ///< clamped_mix with one component
    DS_DIST_EDF_CLAMPED_MIX_2,                       ///< clamped_mix with two components
    DS_DIST_EDF_CLAMPED_MIX_3,                       ///< clamped_mix with three components
    DS_DIST_EDF_UNBOUNDED_MIX_1,                     ///< unbounded_mix with one component
    DS_DIST_EDF_UNBOUNDED_MIX_2,                     ///< unbounded_mix with two components
    DS_DIST_EDF_UNBOUNDED_MIX_3,                     ///< unbounded_mix with three components
    DS_DIST_EDF_COLOR_MIX_1 = 0x10050,               ///< color_normalized_mix with one component
    DS_DIST_EDF_COLOR_MIX_2,                         ///< color_normalized_mix with two components
    DS_DIST_EDF_COLOR_MIX_3,                         ///< color_normalized_mix with three components
    DS_DIST_EDF_COLOR_CLAMPED_MIX_1,                 ///< color_clamped_mix with one component
    DS_DIST_EDF_COLOR_CLAMPED_MIX_2,                 ///< color_clamped_mix with two components
    DS_DIST_EDF_COLOR_CLAMPED_MIX_3,                 ///< color_clamped_mix with three components
    DS_DIST_EDF_COLOR_UNBOUNDED_MIX_1,               ///< color_unbounded_mix with one component
    DS_DIST_EDF_COLOR_UNBOUNDED_MIX_2,               ///< color_unbounded_mix with two components
    DS_DIST_EDF_COLOR_UNBOUNDED_MIX_3,               ///< color_unbounded_mix with three components
    DS_DIST_VDF_MIX_1 = 0x10060,                     ///< normalized_mix with one component
    DS_DIST_VDF_MIX_2,                               ///< normalized_mix with two components
    DS_DIST_VDF_MIX_3,                               ///< normalized_mix with three components
    DS_DIST_VDF_CLAMPED_MIX_1,                       ///< clamped_mix with one component
    DS_DIST_VDF_CLAMPED_MIX_2,                       ///< clamped_mix with two components
    DS_DIST_VDF_CLAMPED_MIX_3,                       ///< clamped_mix with three components
    DS_DIST_VDF_UNBOUNDED_MIX_1,                     ///< unbounded_mix with one component
    DS_DIST_VDF_UNBOUNDED_MIX_2,                     ///< unbounded_mix with two components
    DS_DIST_VDF_UNBOUNDED_MIX_3,                     ///< unbounded_mix with three components
    DS_DIST_VDF_COLOR_MIX_1 = 0x10070,               ///< color_normalized_mix with one component
    DS_DIST_VDF_COLOR_MIX_2,                         ///< color_normalized_mix with two components
    DS_DIST_VDF_COLOR_MIX_3,                         ///< color_normalized_mix with three components
    DS_DIST_VDF_COLOR_CLAMPED_MIX_1,                 ///< color_clamped_mix with one component
    DS_DIST_VDF_COLOR_CLAMPED_MIX_2,                 ///< color_clamped_mix with two components
    DS_DIST_VDF_COLOR_CLAMPED_MIX_3,                 ///< color_clamped_mix with three components
    DS_DIST_VDF_COLOR_UNBOUNDED_MIX_1,               ///< color_unbounded_mix with one component
    DS_DIST_VDF_COLOR_UNBOUNDED_MIX_2,               ///< color_unbounded_mix with two components
    DS_DIST_VDF_COLOR_UNBOUNDED_MIX_3,               ///< color_unbounded_mix with three components
    DS_DIST_MATERIAL_CONDITIONAL_OPERATOR = 0x10100, ///< ternary ?: operator on material type
    DS_DIST_BSDF_CONDITIONAL_OPERATOR,               ///< ternary ?: operator on bsdf type
    DS_DIST_EDF_CONDITIONAL_OPERATOR,                ///< ternary ?: operator on edf type
    DS_DIST_VDF_CONDITIONAL_OPERATOR,                ///< ternary ?: operator on vdf type
    DS_DIST_BSDF_DIRECTIONAL_FACTOR = 0x10110,       ///< directional factor for BSDF
    DS_DIST_EDF_DIRECTIONAL_FACTOR,                  ///< directional factor for EDF
    DS_DIST_LOCAL_NORMAL = 0x11000                   ///< local_normal
};

///
/// An interface for reporting rule matcher events.
///
class IRule_matcher_event {
public:
    /// A DAG path is checked against a rule set.
    ///
    /// \param rule_set_name   the name of the rule set
    /// \param dag_path        the DAG path to a node that is currently checked
    virtual void path_check_event(
        char const *rule_set_name,
        char const *dag_path) = 0;

    /// A rule has matched.
    ///
    /// \param rule_set_name   the name of the rule set
    /// \param rule_id         the rule id
    /// \param rule_name       the name of the rule that matched
    /// \param file_name       if non-NULL, the file name where the rule was declared
    /// \param line_number     if non-ZERO, the line number where the rule was declared
    virtual void rule_match_event(
        char const *rule_set_name,
        unsigned   rule_id,
        char const *rule_name,
        char const *file_name,
        unsigned   line_number) = 0;

    /// A postcondition has failed.
    ///
    /// \param rule_set_name   the name of the rule set
    virtual void postcondition_failed(
        char const *rule_set_name) = 0;

    /// A postcondition has failed for a given path.
    ///
    /// \param path   the path that failed
    virtual void postcondition_failed_path(
        char const *path) = 0;

    /// A rule with an attached debug_print() statement has matched. This function
    /// receives the same arguments as rule_match_event() plus the name of the variable
    /// to be printed and the value of the variable after matching succeeded.
    ///
    /// \param plugin_api      reference to the distiller plugin API that issued this
    ///                        debug_print() event
    /// \param rule_set_name   the name of the rule set
    /// \param rule_id         the rule id
    /// \param rule_name       the name of the rule that matched
    /// \param file_name       if non-NULL, the file name where the rule was declared
    /// \param line_number     if non-ZERO, the line number where the rule was declared
    /// \param var_name        name of the variable whose value is printed
    /// \param value           value to be printed
    virtual void debug_print(
        IDistiller_plugin_api &plugin_api,
        char const *rule_set_name,
        unsigned   rule_id,
        char const *rule_name,
        char const *file_name,
        unsigned   line_number,
        char const *var_name,
        DAG_node const *value) = 0;

};

///
/// The interface to the generated rule matcher, new version for distiller plugin API.
///
class IRule_matcher {
public:
    /// Function pointer for individual checker functions in rule sets.
    typedef bool (*Checker_function)(
        IDistiller_plugin_api &plugin_api,
        DAG_node const        *node);

    /// Return the strategy to be used with this rule set.
    virtual Rule_eval_strategy get_strategy() const = 0;

    /// Return the number of fully qualified imported MDL names in
    /// this rule set.
    virtual size_t get_target_material_name_count() const = 0;

    /// Return the fully qualified name of the imported MDL name with
    /// the given index for this rule set.
    virtual char const *get_target_material_name(size_t i) const = 0;

    /// Run the matcher.
    ///
    /// \param event_handler  if non-NULL, an event handler used to report events
    /// \param plugin_api     the plugin API used to create new nodes
    /// \param node           the root node of the sub-DAG to match
    /// \param options        options to customize constants and behavior in rules
    /// \param result_code    returns a code that controls the matching algorithm
    virtual DAG_node const *matcher(
        IRule_matcher_event     *event_handler,
        IDistiller_plugin_api   &plugin_api,
        DAG_node const          *node,
        mi::mdl::Distiller_options const *options,
        Rule_result_code        &result_code) const = 0;

    /// Returns true if the root node satisfies all postcondition that need to
    /// be valid after the rule set has been run.
    ///
    /// \param event_handler  if non-NULL, an event handler used to report events
    /// \param plugin_api  the plugin API used to create new nodes
    /// \param root        the root node of the DAG to check
    /// \param options     options to customize constants and behavior in rules
    virtual bool postcond(
        IRule_matcher_event     *event_handler,
        IDistiller_plugin_api   &plugin_api,
        DAG_node const          *root,
        mi::mdl::Distiller_options const *options) const = 0;

    /// Return the name of the rule set (for diagnostic messages)
    virtual char const *get_rule_set_name() const = 0;

    /// Set the Node_types object to use.
    /// This is used by the Distiller to install the node type helper
    /// class in the rule object.
    virtual void set_node_types(Node_types *node_types) = 0;
};

} // mdl
} // mi

#endif // MDL_DISTILLER_RULES_H
