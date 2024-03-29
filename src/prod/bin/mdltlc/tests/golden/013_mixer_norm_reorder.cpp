/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
// Generated by mdltlc

#include "pch.h"

#include "013_mixer_norm_reorder.h"

#include <mi/mdl/mdl_distiller_plugin_api.h>
#include <mi/mdl/mdl_distiller_plugin_helper.h>

using namespace mi::mdl;

namespace MI {
namespace DIST {

// Return the strategy to be used with this rule set.
Rule_eval_strategy Mixer_norm_reorder::get_strategy() const {
    return RULE_EVAL_TOP_DOWN;
}

// Return the name of the rule set.
char const * Mixer_norm_reorder::get_rule_set_name() const {
    return "Mixer_norm_reorder";
}

// Return the number of imports of this rule set.
size_t Mixer_norm_reorder::get_target_material_name_count() const {
    return 0;
}

// Return the name of the import at index i.
char const *Mixer_norm_reorder::get_target_material_name(size_t i) const {
    return nullptr;
}

// Run the matcher.
DAG_node const* Mixer_norm_reorder::matcher(
    IRule_matcher_event *event_handler,
    IDistiller_plugin_api &e,
    DAG_node const *node,
    const mi::mdl::Distiller_options *options,
    Rule_result_code &result_code)const
{
    switch (e.get_selector(node)) {
    case mi::mdl::DS_DIST_STRUCT_MATERIAL: // match for material(tw, material_surface(custom_curve_layer(f0_3, f90_3, e3, w3, microfacet_ggx_vcavities_bsdf(ru4, rv4, tint4, _, t4), bsdf_mix_2(w2, custom_curve_layer(f0, f90, e, w, microfacet_ggx_vcavities_bsdf(ru2, rv2, tint2, _, t2), base, n), w1, microfacet_ggx_vcavities_bsdf(ru1, rv1, tint1, _, t1)), n3), em), bf, ior, vol, material_geometry(d, cutout, ng))
// 013_mixer_norm_reorder.mdltl:16
// deadrule RUID 8293
        if (true
        && (e.get_selector(e.get_compound_argument(node, 1)) == mi::mdl::DS_DIST_STRUCT_MATERIAL_SURFACE)
        && (e.get_selector(e.get_compound_argument(e.get_compound_argument(node, 1), 0)) == mi::mdl::IDefinition::Semantics::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        && (e.get_selector(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 4)) == mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
        && (e.get_selector(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5)) == mi::mdl::DS_DIST_BSDF_MIX_2)
        && (e.get_selector(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1)) == mi::mdl::IDefinition::Semantics::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER)
        && (e.get_selector(e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 4)) == mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
        && (e.get_selector(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 3)) == mi::mdl::IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF)
        && (e.get_selector(e.get_compound_argument(node, 5)) == mi::mdl::DS_DIST_STRUCT_MATERIAL_GEOMETRY)) {
            const DAG_node* v_tw = e.get_compound_argument(node, 0);
            const DAG_node* v_f0_3 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 0);
            const DAG_node* v_f90_3 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 1);
            const DAG_node* v_e3 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 2);
            const DAG_node* v_w3 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 3);
            const DAG_node* v_ru4 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 4), 0);
            const DAG_node* v_rv4 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 4), 1);
            const DAG_node* v_tint4 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 4), 2);
            const DAG_node* v_t4 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 4), 4);
            const DAG_node* v_w2 = e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 0);
            const DAG_node* v_f0 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 0);
            const DAG_node* v_f90 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 1);
            const DAG_node* v_e = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 2);
            const DAG_node* v_w = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 3);
            const DAG_node* v_ru2 = e.get_compound_argument(e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 4), 0);
            const DAG_node* v_rv2 = e.get_compound_argument(e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 4), 1);
            const DAG_node* v_tint2 = e.get_compound_argument(e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 4), 2);
            const DAG_node* v_t2 = e.get_compound_argument(e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 4), 4);
            const DAG_node* v_base = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 5);
            const DAG_node* v_n = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 1), 6);
            const DAG_node* v_w1 = e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 2);
            const DAG_node* v_ru1 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 3), 0);
            const DAG_node* v_rv1 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 3), 1);
            const DAG_node* v_tint1 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 3), 2);
            const DAG_node* v_t1 = e.get_compound_argument(e.get_remapped_argument(e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 5), 3), 4);
            const DAG_node* v_n3 = e.get_compound_argument(e.get_compound_argument(e.get_compound_argument(node, 1), 0), 6);
            const DAG_node* v_em = e.get_compound_argument(e.get_compound_argument(node, 1), 1);
            const DAG_node* v_bf = e.get_compound_argument(node, 2);
            const DAG_node* v_ior = e.get_compound_argument(node, 3);
            const DAG_node* v_vol = e.get_compound_argument(node, 4);
            const DAG_node* v_d = e.get_compound_argument(e.get_compound_argument(node, 5), 0);
            const DAG_node* v_cutout = e.get_compound_argument(e.get_compound_argument(node, 5), 1);
            const DAG_node* v_ng = e.get_compound_argument(e.get_compound_argument(node, 5), 2);
            if (e.eval_maybe_if(e.create_binary(
                IDistiller_plugin_api::OK_NOT_EQUAL,
                    v_ru2,
                    v_ru1))) {
                if (event_handler != nullptr)
                    fire_match_event(*event_handler, 0);
                result_code = RULE_SKIP_RECURSION;
                return e.create_call("material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)",
                    IDefinition::DS_ELEM_CONSTRUCTOR, Args_wrapper<7>::mk_args(e,m_node_types,
                        material, v_tw, e.create_call("material_surface(bsdf,material_emission)",
                            IDefinition::DS_ELEM_CONSTRUCTOR, Args_wrapper<2>::mk_args(
                                e,m_node_types, material_surface, e.create_call("::df::custom_curve_layer(float,float,float,float,bsdf,bsdf,float3)",
                                    IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER,
                                    Args_wrapper<7>::mk_args(e,m_node_types, custom_curve_layer,
                                        v_f0_3, v_f90_3, v_e3, v_w3, e.create_call("::df::microfacet_ggx_vcavities_bsdf(float,float,color,color,float3,::df::scatter_mode,string)",
                                            IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
                                            Args_wrapper<7>::mk_args(e,m_node_types,
                                                microfacet_ggx_vcavities_bsdf, v_ru4,
                                                v_rv4, v_tint4, e.create_color_constant(0,0,0),
                                                v_t4).args, 7, e.get_type_factory()->create_bsdf()),
                                        e.create_mixer_call(Args_wrapper<4>::mk_args(e,m_node_types,
                                                node_null, v_w1, e.create_call("::df::microfacet_ggx_vcavities_bsdf(float,float,color,color,float3,::df::scatter_mode,string)",
                                                    IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
                                                    Args_wrapper<7>::mk_args(e,m_node_types,
                                                        microfacet_ggx_vcavities_bsdf,
                                                        e.create_function_call("::nvidia::distilling_support::average",
                                                            Nodes_wrapper<4>(v_w1,
                                                                v_ru1, v_w2, v_ru2).data(),
                                                            4), e.create_function_call("::nvidia::distilling_support::average",
                                                            Nodes_wrapper<4>(v_w1,
                                                                v_rv1, v_w2, v_rv2).data(),
                                                            4), v_tint1, e.create_color_constant(0,0,0),
                                                        v_t1).args, 7, e.get_type_factory()->create_bsdf()),
                                                v_w2, e.create_call("::df::custom_curve_layer(float,float,float,float,bsdf,bsdf,float3)",
                                                    IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER,
                                                    Args_wrapper<7>::mk_args(e,m_node_types,
                                                        custom_curve_layer, v_f0,
                                                        v_f90, v_e, v_w, e.create_call("::df::microfacet_ggx_vcavities_bsdf(float,float,color,color,float3,::df::scatter_mode,string)",
                                                            IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF,
                                                            Args_wrapper<7>::mk_args(
                                                                e,m_node_types, microfacet_ggx_vcavities_bsdf,
                                                                e.create_function_call("::nvidia::distilling_support::average",
                                                                    Nodes_wrapper<4>(v_w1,
                                                                        v_ru1, v_w2,
                                                                        v_ru2).data(),
                                                                    4), e.create_function_call("::nvidia::distilling_support::average",
                                                                    Nodes_wrapper<4>(v_w1,
                                                                        v_rv1, v_w2,
                                                                        v_rv2).data(),
                                                                    4), v_tint2,
                                                                e.create_color_constant(0,0,0),
                                                                v_t2).args, 7, e.get_type_factory()->create_bsdf()),
                                                        v_base, v_n).args, 7, e.get_type_factory()->create_bsdf())).args,
                                            4), v_n3).args, 7, e.get_type_factory()->create_bsdf()),
                                v_em).args, 2, e.get_type_factory()->get_predefined_struct(
                            IType_struct::SID_MATERIAL_SURFACE)), v_bf, v_ior, v_vol,
                        e.create_call("material_geometry(float3,float,float3)", IDefinition::DS_ELEM_CONSTRUCTOR,
                            Args_wrapper<3>::mk_args(e,m_node_types, material_geometry,
                                v_d, v_cutout, v_ng).args, 3, e.get_type_factory()->
                            get_predefined_struct(IType_struct::SID_MATERIAL_GEOMETRY))).args,
                    7, e.get_type_factory()->get_predefined_struct(IType_struct::SID_MATERIAL));
            }
        }
        break;
    default:
        break;
    }

    return node;
}

bool Mixer_norm_reorder::postcond(
    IRule_matcher_event *event_handler,
    IDistiller_plugin_api &e,
    DAG_node const *root,
    const mi::mdl::Distiller_options *options) const
{
    (void)e; (void)root; // no unused variable warnings
    bool result = true;
    if (!result && event_handler != NULL)
        fire_postcondition_event(*event_handler);
    return result;
}

void Mixer_norm_reorder::fire_match_event(
    mi::mdl::IRule_matcher_event &event_handler,
    std::size_t id)
{
    Rule_info const &ri = g_rule_info[id];
    event_handler.rule_match_event("Mixer_norm_reorder", ri.ruid, ri.rname, ri.fname,
        ri.fline);
}

void Mixer_norm_reorder::fire_postcondition_event(
mi::mdl::IRule_matcher_event &event_handler)
{
    event_handler.postcondition_failed("Mixer_norm_reorder");
}

void Mixer_norm_reorder::fire_debug_print(
    mi::mdl::IDistiller_plugin_api &plugin_api,
    mi::mdl::IRule_matcher_event &event_handler,
    std::size_t idx,
    char const *var_name,
    DAG_node const *value)
{
    Rule_info const &ri = g_rule_info[idx];
    event_handler.debug_print(plugin_api, "Mixer_norm_reorder", ri.ruid, ri.rname,
        ri.fname, ri.fline, var_name, value);
}


// Rule info table.
Mixer_norm_reorder::Rule_info const Mixer_norm_reorder::g_rule_info[1] = {
    { 8293, "material", "013_mixer_norm_reorder.mdltl", 16 }
};


} // DIST
} // MI
// End of generated code
