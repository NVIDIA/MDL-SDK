/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_distiller.h"

#include <string>

#include <mi/base/ilogger.h>
#include <mi/neuraylib/ilogging_configuration.h>
#include <base/system/version/i_version.h>

#include <mi/mdl/mdl_distiller_plugin_api.h>

#include <iostream>

#include "dist_rules.h"
#include "dist_rules_ue.h"
#include "dist_rules_transmissive_pbr.h"

namespace MI {
namespace DIST {

// Supported distilling targets in this plugin
// Note: these need to match the switch( target_index) statement below
static const char* s_targets[] = {
    "_dbg_simple",
    "diffuse",
    "specular_glossy",
    "ue4",
    "transmissive_pbr"
};

/// Returns the dimension of an array.
template<typename T, size_t n>
inline size_t dimension_of(T (&c)[n]) { return n; }


// Global logger
mi::base::Handle<mi::base::ILogger> g_logger;

// Support function to log messages. Skips if no logger is available
void log( mi::base::Message_severity severity, const char* message)
{
    if( g_logger.is_valid_interface())
        g_logger->message( severity, "DISTILLER:COMPILER", message);
}

static bool uses_ternary_df(
    const mi::mdl::IMaterial_instance* material_instance)
{
    return 0 != (material_instance->get_properties()
        & mi::mdl::IMaterial_instance::IP_USES_TERNARY_OPERATOR_ON_DF);
}

bool Mdl_distiller::init( mi::base::ILogger* logger) {
    g_logger = mi::base::make_handle_dup(logger);

    std::string message = "Plugin \"";
    message += MI_DISTILLER_PLUGIN_NAME;
    message += "\" (build " + std::string( MI::VERSION::get_platform_version());
    message += ", " + std::string( MI::VERSION::get_platform_date());
    message += ") initialized";
    log( mi::base::MESSAGE_SEVERITY_INFO, message.c_str());

    return true;
}

bool Mdl_distiller::exit() {
    g_logger = 0;
    return true;
}

/// Returns the number of available distilling targets.
mi::Size Mdl_distiller::get_target_count() const {
    return dimension_of( s_targets);
}

/// Returns the name of the distilling target at index position.
const char* Mdl_distiller::get_target_name( mi::Size index) const {
    return (index < dimension_of( s_targets)) ? s_targets[index] : nullptr;
}

mi::Size Mdl_distiller::get_required_module_count(mi::Size target_index) const {
    return 0;
}

const char *Mdl_distiller::get_required_module_code(mi::Size target_index, mi::Size index) const {
    return nullptr;
}

const char *Mdl_distiller::get_required_module_name(mi::Size target_index, mi::Size index) const {
    return nullptr;
}

/// Main function to distill an MDL material.
const mi::mdl::IMaterial_instance* Mdl_distiller::distill(
    mi::mdl::IDistiller_plugin_api&    api,
    mi::mdl::IRule_matcher_event*      event_handler,
    const mi::mdl::IMaterial_instance* material_instance,
    mi::Size                           target_index,
    mi::mdl::Distiller_options*        options,
    mi::Sint32*                        p_error) const
{
    // Switch from error pointer to reference to simplify later code for the case of p_error == 0.
    mi::Sint32 dummy;
    mi::Sint32 &error = p_error != NULL ? *p_error : dummy;
    error = 0;

    // Check the preconditions on the input parameters
    if ( (!material_instance) || (!options) || (target_index >= get_target_count())) {
        error = -3;
        return nullptr;
    }

    mi::base::Handle<mi::mdl::IMaterial_instance const> res;

#define CHECK_RESULT  if(error != 0) { return NULL; }

    // Note: the case numbers must match the cardinal order of targets in s_targets above
    switch ( target_index) {
    case 0:  // "_dbg_simple"
    {
        log( mi::base::MESSAGE_SEVERITY_INFO, "Distilling to target '_dbg_simple'.");
        Make_simple_rules make_simple;
        res = api.apply_rules( material_instance, make_simple, event_handler, options, error);
        break;
    }
    case 1:  // "diffuse"
    {
        log( mi::base::MESSAGE_SEVERITY_INFO, "Distilling to target 'diffuse'.");
        res = mi::base::make_handle_dup(material_instance);
        if ( uses_ternary_df( material_instance)) {
            Elide_conditional_operator_rules cond_operator;
            res = api.apply_rules( res.get(), cond_operator, event_handler, options, error);
            CHECK_RESULT;
        }
        Make_simple_rules make_simple;
        res = api.apply_rules( res.get(), make_simple, event_handler, options, error);
        CHECK_RESULT;

        Elide_tint_rules elide_tint;
        res = api.apply_rules( res.get(), elide_tint, event_handler, options, error);
        CHECK_RESULT;

        if ( options->layer_normal) {
            Make_normal_rules make_normal;
            res = api.apply_rules( res.get(), make_normal, event_handler, options, error);
            CHECK_RESULT
        }

        Elide_layering_rules elide_layering;
        res = api.apply_rules( res.get(), elide_layering, event_handler, options, error);
        CHECK_RESULT

        Make_diffuse_rules make_diffuse;
        res = api.apply_rules( res.get(), make_diffuse, event_handler, options, error);
        CHECK_RESULT;
        break;
    }
    case 2: // "specular_glossy"
    {
        //without coat for now
        log( mi::base::MESSAGE_SEVERITY_INFO,
                            "Distilling to target 'spec glossiness'.");
        Reduce_1_4_to_1_3_rules make_1_3;
        res = api.apply_rules( material_instance, make_1_3, event_handler, options, error);
        CHECK_RESULT;

        if ( uses_ternary_df(material_instance)) {
            Elide_conditional_operator_rules cond_operator;
            res = api.apply_rules( res.get(), cond_operator, event_handler, options, error);
            CHECK_RESULT;
        }
        Make_simple_for_ue4 make_simple;
        res = api.apply_rules( res.get(), make_simple, event_handler, options, error);
        CHECK_RESULT;

        if ( options->layer_normal) {
            Make_normal_for_sg make_normal_sg;
            res = api.apply_rules( res.get(), make_normal_sg, event_handler, options, error);
            CHECK_RESULT;
        }

        Elide_weighted_layer_for_ue4 elide_layering;
        res = api.apply_rules( res.get(), elide_layering, event_handler, options, error);
        CHECK_RESULT;

        Make_transmission_into_cutout_ue4 make_cutout;
        mi::base::Handle<mi::mdl::IMaterial_instance const> clone2;
        clone2 =  api.apply_rules( res.get(), make_cutout, event_handler, options, error);
        CHECK_RESULT;

        Elide_transmission1 elide_transmission1;
        res = api.apply_rules( res.get(), elide_transmission1, event_handler, options, error);
        CHECK_RESULT;

        Elide_transmission2 elide_transmission2;
        res = api.apply_rules( res.get(), elide_transmission2, event_handler, options, error);
        CHECK_RESULT;

        Elide_tint_for_ue4 elide_tint;
        res = api.apply_rules( res.get(), elide_tint, event_handler, options, error);
        CHECK_RESULT;

        // make_mix_nodes_canonical
        res = api.normalize_mixers( res.get(), event_handler, options, error);
        CHECK_RESULT;

        Make_for_sg make_sg;
        res = api.apply_rules( res.get(), make_sg, event_handler, options, error);
        CHECK_RESULT;

        // if ( options->layer_normal) {
        //     // TBD: check if this is enough or if we need to do something in the else clause
        //     // push the base normal in
        //     res = api.merge_materials( res.get(), clone.get(),
        //                              mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_NORMAL);
        //     CHECK_RESULT;
        // }


        // workaround: merge_materials does not correctly work and overwrites geometry.normal,
        // save normal and restore it later
        Save_normal save_normal;
        res = api.apply_rules( res.get(), save_normal, event_handler, options, error);
        CHECK_RESULT;

        res = api.merge_materials( res.get(), clone2.get(),
                          mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_CUTOUT_OPACITY);
        CHECK_RESULT;

        Restore_normal restore_normal;
        res = api.apply_rules( res.get(), restore_normal, event_handler, options, error);
        CHECK_RESULT;

        Fix_backface fix_backface;
        res = api.apply_rules( res.get(), fix_backface, event_handler, options, error);
        CHECK_RESULT;
        break;
    }
    case 3: // "ue4"
    {
        log( mi::base::MESSAGE_SEVERITY_INFO, "Distilling to target 'ue4'.");
        Reduce_1_4_to_1_3_rules make_1_3;
        res = api.apply_rules( material_instance, make_1_3, event_handler, options, error);
        CHECK_RESULT;

        if ( uses_ternary_df(material_instance)) {
            Elide_conditional_operator_rules cond_operator;
            res = api.apply_rules( res.get(), cond_operator, event_handler, options, error);
            CHECK_RESULT;
        }
        Make_simple_for_ue4 make_simple;
        res = api.apply_rules( res.get(), make_simple, event_handler, options, error);
        CHECK_RESULT;

        //handle hacky materials that use a high dielectric ior
        Adapt_layering_for_ue4 adapt_layering;
        res = api.apply_rules( res.get(), adapt_layering, event_handler, options, error);
        CHECK_RESULT;

        mi::base::Handle<mi::mdl::IMaterial_instance const> clone;
        if ( options->layer_normal) {
            Make_normal_for_ue4 make_normal;
            clone = api.apply_rules( res.get(), make_normal, event_handler, options, error);
           CHECK_RESULT;
        }

        Elide_weighted_layer_for_ue4 elide_layering;
        res = api.apply_rules( res.get(), elide_layering, event_handler, options, error);
        CHECK_RESULT;

        Make_transmission_into_cutout_ue4 make_cutout;
        mi::base::Handle<mi::mdl::IMaterial_instance const> clone2;
        clone2 =  api.apply_rules( res.get(), make_cutout, event_handler, options, error);
        CHECK_RESULT;
        res = api.merge_materials( res.get(), clone2.get(),
                             mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_CUTOUT_OPACITY);
        CHECK_RESULT;

        Elide_transmission1 elide_transmission1;
        res = api.apply_rules( res.get(), elide_transmission1, event_handler, options, error);
        CHECK_RESULT;
        Elide_transmission2 elide_transmission2;
        res = api.apply_rules( res.get(), elide_transmission2, event_handler, options, error);
        CHECK_RESULT;

        Elide_tint_for_ue4 elide_tint;
        res = api.apply_rules( res.get(), elide_tint, event_handler, options, error);
        CHECK_RESULT;
        // make_mix_nodes_canonical
        res = api.normalize_mixers( res.get(), event_handler, options, error);
        CHECK_RESULT;
        Make_for_ue4 make_ue4;
        res = api.apply_rules( res.get(), make_ue4, event_handler, options, error);
        CHECK_RESULT;

        if ( options->layer_normal) {
            // TBD: check if this is enough or if we need to do something in the else clause
            // push the base normal in
            res = api.merge_materials( res.get(), clone.get(),
                                   mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_NORMAL);
            CHECK_RESULT;
        }
        if ( options->merge_metal_and_base_color) {
            Fix_common_tint_for_UE4 fix_common_tint_for_UE4;
            res = api.apply_rules( res.get(), fix_common_tint_for_UE4,
                                  event_handler, options, error);
            CHECK_RESULT;
        }
        Fix_common_roughness_for_UE4 fix_common_roughness_for_UE4;
        res = api.apply_rules( res.get(), fix_common_roughness_for_UE4,
                                  event_handler, options, error);
        CHECK_RESULT;
        Fix_normals_for_UE4 fix_normals_for_ue4;
        res = api.apply_rules( res.get(), fix_normals_for_ue4, event_handler, options, error);
        CHECK_RESULT;
        // restore geometry normal from other copy
        res = api.merge_materials( res.get(), clone2.get(),
                                   mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_NORMAL);
        CHECK_RESULT;
        // eliminate geometry normal altogether so we only have clearcoat and underclearcoat left
        Merge_normals_for_UE4 merge_normals_for_ue4;
        res = api.apply_rules( res.get(), merge_normals_for_ue4, event_handler, options, error);
        CHECK_RESULT;
        Fix_backface fix_backface;
        res = api.apply_rules( res.get(), fix_backface, event_handler, options, error);
        CHECK_RESULT;
        break;
    }
    case 4: // "transmissive_pbr"
    {
        log( mi::base::MESSAGE_SEVERITY_INFO, "Distilling to target 'transmissive_pbr'.");
        Reduce_1_4_to_1_3_rules make_1_3;
        res = api.apply_rules( material_instance, make_1_3, event_handler, options, error);
        CHECK_RESULT;

        if ( uses_ternary_df(material_instance)) {
            Elide_conditional_operator_rules cond_operator;
            res = api.apply_rules( res.get(), cond_operator, event_handler, options, error);
            CHECK_RESULT;
        }
        Make_simple_for_tpbr make_simple;
        res = api.apply_rules( res.get(), make_simple, event_handler, options, error);
        CHECK_RESULT;

        //handle hacky materials that use a high dielectric ior
        Adapt_layering_for_ue4 adapt_layering;
        res = api.apply_rules( res.get(), adapt_layering, event_handler, options, error);
        CHECK_RESULT;

        mi::base::Handle<mi::mdl::IMaterial_instance const> clone;
        if ( options->layer_normal) {
            Make_normal_for_tpbr make_normal;
            clone = api.apply_rules( res.get(), make_normal, event_handler, options, error);
           CHECK_RESULT;
        }
        //still need the clone for normal

        Elide_weighted_layer_for_ue4 elide_layering;
        res = api.apply_rules( res.get(), elide_layering, event_handler, options, error);
        CHECK_RESULT;


        Collect_transmission_for_tpbr collect_transmission;
        mi::base::Handle<mi::mdl::IMaterial_instance const> clone2;
        clone2 =  api.apply_rules( res.get(), collect_transmission, event_handler, options, error);
        CHECK_RESULT;

        //move transmission result into the backface
        Fix_backface fix_backface;
        clone2 = api.apply_rules( clone2.get(), fix_backface, event_handler, options, error);
        CHECK_RESULT;

        Elide_transmission_for_tpbr elide_transmission;
        res = api.apply_rules( res.get(), elide_transmission, event_handler, options, error);
        CHECK_RESULT;

        Elide_tint_for_tpbr elide_tint;
        res = api.apply_rules( res.get(), elide_tint, event_handler, options, error);
        CHECK_RESULT;

        // make_mix_nodes_canonical
        res = api.normalize_mixers( res.get(), event_handler, options, error);
        CHECK_RESULT;
        Make_for_ue4 make_ue4;
        res = api.apply_rules( res.get(), make_ue4, event_handler, options, error);
        CHECK_RESULT;

        if ( options->layer_normal) {
            // TBD: check if this is enough or if we need to do something in the else clause
            // push the base normal in
            res = api.merge_materials( res.get(), clone.get(),
                                     mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_NORMAL);
            CHECK_RESULT;
        }
        if ( options->merge_metal_and_base_color) {
            Fix_common_tint_for_UE4 fix_common_tint_for_UE4;
            res = api.apply_rules( res.get(), fix_common_tint_for_UE4,
                                      event_handler, options, error);
            CHECK_RESULT;
        }
        Fix_common_roughness_for_tpbr fix_common_roughness_for_tpbr;
        res = api.apply_rules( res.get(), fix_common_roughness_for_tpbr,
                                  event_handler, options, error);
        CHECK_RESULT;
        Fix_normals_for_UE4 fix_normals_for_ue4;
        res = api.apply_rules( res.get(), fix_normals_for_ue4, event_handler, options, error);
        CHECK_RESULT;
        // restore geometry normal from other copy
        res = api.merge_materials( res.get(), clone2.get(),
                                   mi::mdl::IDistiller_plugin_api::FS_MATERIAL_GEOMETRY_NORMAL);
        CHECK_RESULT;
        // eliminate geometry normal altogether so we only have clearcoat and underclearcoat left
        Merge_normals_for_UE4 merge_normals_for_ue4;
        res = api.apply_rules( res.get(), merge_normals_for_ue4, event_handler, options, error);
        CHECK_RESULT;

        //insert transmission as necessary
        // transmissivity is stored in backfacescattering of clone2
        res = api.merge_materials( res.get(), clone2.get(),
                                mi::mdl::IDistiller_plugin_api::FS_MATERIAL_BACKFACE_SCATTERING);

        Insert_transmission_for_tpbr insert_transmission_for_tpbr;
        res = api.apply_rules( res.get(), insert_transmission_for_tpbr,
                                  event_handler, options, error);
        CHECK_RESULT;
        if ( options->merge_transmission_and_base_color && options->merge_metal_and_base_color) {
            Fix_common_tint_for_tpbr fix_common_tint_for_tpbr;
            res = api.apply_rules( res.get(), fix_common_tint_for_tpbr,
                                      event_handler, options, error);
            CHECK_RESULT;
        }
        else if ( options->merge_transmission_and_base_color ) {
            Fix_common_tint_2_for_tpbr fix_common_tint_for_tpbr;
            res = api.apply_rules( res.get(), fix_common_tint_for_tpbr,
                                      event_handler, options, error);
            CHECK_RESULT;
        }

        res = api.apply_rules( res.get(), fix_backface, event_handler, options, error);
        CHECK_RESULT;
        break;
    }

    } // end switch

#undef CHECK_RESULT

    if (res.is_valid_interface()) {
        res->retain();
        return res.get();
    }

    error = -3;
    return nullptr;
}


/// Factory to create an instance of Mdl_distiller.
extern "C"
MI_DLL_EXPORT
mi::base::Plugin* mi_plugin_factory(
    mi::Sint32 index,         // index of the plugin
    void* context)            // context given to the library, ignore
{
    if( index > 0)
        return 0;
    return new Mdl_distiller();
}

} // namespace DIST
} // namespace MI
