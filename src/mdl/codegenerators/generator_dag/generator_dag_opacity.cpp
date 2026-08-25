/******************************************************************************
 * Copyright (c) 2015-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "generator_dag_generated_dag.h"
#include "generator_dag_tools.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"
#include "mdl/compiler/stdmodule/enums.h"

namespace mi {
namespace mdl {

namespace {

/// Helper class to check the opacity of a material instance.
class Opacity_analyzer {

public:
    typedef IMaterial_instance::Opacity Result;

    /// Constructor.
    ///
    /// \param material  the material instance construction
    Opacity_analyzer(
        DAG_call const *material)
    : m_constructor(material)
    {
    }

    /// Get the cutout opacity of the material instance if it is constant, NULL otherwise.
    IValue_float const *get_cutout_opacity()
    {
        // must be a material constructor
        MDL_ASSERT(m_constructor->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR);

        // first check the cutout_opacity
        static char const * const path[] = { "geometry", "cutout_opacity" };
        IValue const *v = get_value(path);

        if (v == NULL) {
            // cannot analyze
            return NULL;
        }
        return cast<IValue_float>(v);
    }

    /// Analyze if the given instance is opaque or transparent.
    ///
    /// \param skip_cutout   if true, the analysis of cutout opacity is skipped.
    /// \returns opaque      if the material instance has an opacity of 1.0
    ///          transparent if the material instance has an opacity < 1.0
    ///          unknown     otherwise (might depend on parameters)
    Result analyze(bool skip_cutout)
    {
        if (!skip_cutout) {
            IValue_float const *f_value = get_cutout_opacity();
            if (f_value == NULL) {
                // cannot analyze
                return IMaterial_instance::OPACITY_UNKNOWN;
            }
            if (f_value->get_value() < 1.0f) {
                // not opaque
                return IMaterial_instance::OPACITY_TRANSPARENT;
            }
        }
        // We do not allow different transmission of front and back-side of an MDL material.
        // Hence it is enough to analyze the front-side.
        DAG_node const *frontside = skip_temp(m_constructor->get_argument("surface"));
        if (is<DAG_constant>(frontside)) {
            // only ONE invalid BSDF, this IS opaque
            return IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *fs = as<DAG_call>(frontside);
        if (fs == NULL) {
            // a parameter, cannot decide
            return IMaterial_instance::OPACITY_UNKNOWN;
        }

        DAG_node const *scattering = skip_temp(fs->get_argument("scattering"));
        if (is<DAG_constant>(scattering)) {
            // only ONE invalid BSDF, this IS opaque
            return IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *sc = as<DAG_call>(scattering);
        if (sc == NULL) {
            // a parameter, cannot decide
            return IMaterial_instance::OPACITY_UNKNOWN;
        }

        return analyze_bsdf(sc);
    }

private:
    /// Skip a temporary.
    ///
    /// \param expr  the DAG node
    ///
    /// \return expr if the node is not a temporary, its value otherwise
    DAG_node const *skip_temp(DAG_node const *expr)
    {
        if (DAG_temporary const *temp = as<DAG_temporary>(expr)) {
            expr = temp->get_expr();
        }
        return expr;
    }

    /// Analyze if a bsdf mixer is opaque or transparent.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_bsdf_mixer(DAG_call const *bsdf)
    {
        DAG_node const *components = skip_temp(bsdf->get_argument("components"));
        if (is<DAG_constant>(components)) {
            // can contain only invalid refs
            return IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *arr = as<DAG_call>(components);
        if (arr == NULL) {
            // a parameter, cannot decide
            return IMaterial_instance::OPACITY_UNKNOWN;
        }
        if (arr->get_semantic() != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
            // not an array constructor, unsupported
            return IMaterial_instance::OPACITY_UNKNOWN;
        }

        bool first = true;
        Result first_result = IMaterial_instance::OPACITY_OPAQUE;

        int n = arr->get_argument_count();
        for (int i = 0; i < n; ++i) {
            DAG_node const *elem = skip_temp(arr->get_argument(i));

            if (is<DAG_constant>(elem)) {
                // can contain only invalid refs
                continue;
            }
            DAG_call const *elem_const = as<DAG_call>(elem);
            if (elem_const == NULL) {
                // a parameter, cannot decide
                return IMaterial_instance::OPACITY_UNKNOWN;
            }
            if (elem_const->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
                // not a struct constructor, cannot decide
                return IMaterial_instance::OPACITY_UNKNOWN;
            }

            DAG_node const *w = elem_const->get_argument("weight");
            if (is_zero_const(w)) {
                // filter out zero components, these will not add anything, ignore them
                continue;
            }

            DAG_node const *bsdf = elem_const->get_argument("component");
            Result res = analyze_bsdf(bsdf);
            if (first) {
                first_result = res;
                first = false;
            } else if (res != first_result) {
                // different components have different opacity, cannot decide
                return IMaterial_instance::OPACITY_UNKNOWN;
            }
        }

        // all identical
        return first_result;
    }

    /// Check if the given DAG IR node represents a call to state::normal().
    static bool is_state_normal_call(DAG_node const *node)
    {
        if (is<DAG_call>(node)) {
            return cast<DAG_call>(node)->get_semantic() == IDefinition::DS_INTRINSIC_STATE_NORMAL;
        }
        return false;
    }

    /// Check if the given DAG IR node is non-null and represents a zero constant.
    static bool is_zero_const(DAG_node const *node)
    {
        if (node != NULL && is<DAG_constant>(node)) {
            IValue const *v = cast<DAG_constant>(node)->get_value();
            return v->is_zero();
        }
        return false;
    }

    /// Check if the given DAG IR node is non-null and represents a one constant.
    static bool is_one_const(DAG_node const *node)
    {
        if (node != NULL && is<DAG_constant>(node)) {
            IValue const *v = cast<DAG_constant>(node)->get_value();
            return v->is_one();
        }
        return false;
    }

    /// Analyze if a bsdf layerer is opaque or transparent.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_bsdf_layerer(DAG_call const *bsdf)
    {
        // Note: we cannot assume that all optimizations are applied to the DAG,
        // so we try to handle some special cases here. This is especially usefull for the
        // distiller, where the structure of the *df part is "shaped" by the distiller rules.
        DAG_node const *weight = skip_temp(bsdf->get_argument("weight"));
        if (is<DAG_constant>(weight)) {
            IValue const *v = cast<DAG_constant>(weight)->get_value();
            if (v->is_zero()) {
                // df::fresnel_layer(weight: 0.0, base: x) ==> x
                // df::color_fresnel_layer(weight: color(0.0), base: x) ==> x
                // df::weighted_layer(weight: 0.0f, base: x) ==> x
                // df::custom_curve_layer(weight: 0.0, base: x) ==> x
                // df::color_weighted_layer(weight: color(0.0f), base: x) ==> x
                // df::custom_curve_layer(weight: color(0.0), base: x) ==> x
                switch (bsdf->get_semantic()) {
                case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
                case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
                case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
                case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
                case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
                case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
                    {
                        DAG_node const *base = skip_temp(bsdf->get_argument("base"));
                        return analyze_bsdf(base);
                    }
                    break;
                default:
                    break;
                }
            } else if (v->is_one()) {
                switch (bsdf->get_semantic()) {
                case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
                case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
                    {
                        // df::weighted_layer(
                        //      weight: 1.0f, layer: x, normal: state::normal()) ==> x
                        // df::color_weighted_layer(
                        //      weight: color(1.0f), layer: x, normal: state::normal()) ==> x
                        DAG_node const *normal = skip_temp(bsdf->get_argument("normal"));
                        if (is_state_normal_call(normal)) {
                            DAG_node const *layer = skip_temp(bsdf->get_argument("layer"));
                            return analyze_bsdf(layer);
                        }
                    }
                    break;
                case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
                    {
                        DAG_node const *normal = skip_temp(bsdf->get_argument("normal"));
                        if (is_state_normal_call(normal)) {
                            DAG_node const *exponent = skip_temp(bsdf->get_argument("exponent"));
                            if (is_zero_const(exponent)) {
                                // df::custom_curve_layer(
                                //      weight: 1.0,
                                //      exponent: 0.0,
                                //      layer: x,
                                //      normal: state::normal()) ==> x
                                DAG_node const *layer = skip_temp(bsdf->get_argument("layer"));
                                return analyze_bsdf(layer);
                            }

                            DAG_node const *normal_reflectivity  =
                                skip_temp(bsdf->get_argument("normal_reflectivity"));
                            DAG_node const *grazing_reflectivity =
                                skip_temp(bsdf->get_argument("grazing_reflectivity"));
                            if (is_one_const(normal_reflectivity) &&
                                is_one_const(grazing_reflectivity))
                            {
                                // df::custom_curve_layer(
                                //      weight: 1.0,
                                //      normal_reflectivity: 1.0,
                                //      grazing_reflectivity: 1.0,
                                //      layer: x,
                                //      normal: state::normal()) ==> x
                                DAG_node const *layer = skip_temp(bsdf->get_argument("layer"));
                                return analyze_bsdf(layer);
                            }
                        }
                    }
                    break;
                case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
                    {
                        DAG_node const *normal     = skip_temp(bsdf->get_argument("normal"));
                        DAG_node const *f82_factor = skip_temp(bsdf->get_argument("f82_factor"));
                        if (is_state_normal_call(normal) && is_one_const(f82_factor)) {
                            DAG_node const *exponent = skip_temp(bsdf->get_argument("exponent"));
                            if (is_zero_const(exponent)) {
                                // df::color_custom_curve_layer(
                                //      weight: color(1.0),
                                //      f82_factor: color(1.0)
                                //      exponent: 0.0,
                                //      layer: x,
                                //      normal: state::normal()) ==> x
                                DAG_node const *layer = skip_temp(bsdf->get_argument("layer"));
                                return analyze_bsdf(layer);
                            }
                            DAG_node const *normal_reflectivity  =
                                skip_temp(bsdf->get_argument("normal_reflectivity"));
                            DAG_node const *grazing_reflectivity =
                                skip_temp(bsdf->get_argument("grazing_reflectivity"));
                            if (is_one_const(normal_reflectivity) &&
                                is_one_const(grazing_reflectivity))
                            {
                                // df::color_custom_curve_layer(
                                //      weight: color(1.0),
                                //      f82_factor: color(1.0)
                                //      normal_reflectivity: 1.0,
                                //      grazing_reflectivity: 1.0,
                                //      layer: x,
                                //      normal: state::normal()) ==> x
                                DAG_node const *layer = skip_temp(bsdf->get_argument("layer"));
                                return analyze_bsdf(layer);
                            }
                        }
                    }
                    break;
                default:
                    break;
                }
            }
        }

        DAG_node const *lower_layer = skip_temp(bsdf->get_argument("base"));
        DAG_node const *upper_layer = skip_temp(bsdf->get_argument("layer"));

        Result low_res = analyze_bsdf(lower_layer);
        Result up_res  = analyze_bsdf(upper_layer);

        if (low_res == up_res)
            return low_res;
        return IMaterial_instance::OPACITY_UNKNOWN;
    }

    /// Analyze if a bsdf modifier is opaque or transparent.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_bsdf_modifier(DAG_call const *bsdf)
    {
        DAG_node const *base = skip_temp(bsdf->get_argument("base"));
        return analyze_bsdf(base);
    }

    /// Analyze if a glossy bsdf is opaque or transparent.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_glossy_bsdf(DAG_call const *bsdf)
    {
        // MaterialLayerBSDF_DBSDF
        df::scatter_mode refl_type = df::scatter_reflect;
        bool has_mode = false;

        switch (bsdf->get_semantic()) {
        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:

            has_mode = true;
            break;


        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:
            break;

        default:
            MDL_ASSERT(!"unhandled glossy BSDF");
            return IMaterial_instance::OPACITY_UNKNOWN;
        }

        if (has_mode) {
            IValue const *v = get_value(bsdf, "mode");
            if (v == NULL) {
                return IMaterial_instance::OPACITY_UNKNOWN;
            }
            IValue_int_valued const *i_v = cast<IValue_int_valued>(v);
            refl_type = static_cast<df::scatter_mode>(i_v->get_value());
        }

        if (refl_type == df::scatter_transmit || refl_type == df::scatter_reflect_transmit) {
            return IMaterial_instance::OPACITY_TRANSPARENT;
        }
        return IMaterial_instance::OPACITY_OPAQUE;
    }

    /// Analyze if an elemental bsdf is opaque or transparent.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_elemental_bsdf(DAG_call const *bsdf)
    {
        switch (bsdf->get_semantic()) {
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF:
            // MaterialLayerBSDF_DiffuseRefl
            return IMaterial_instance::OPACITY_OPAQUE;

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
            // MaterialLayerBSDF_DiffuseTrans;
            return IMaterial_instance::OPACITY_TRANSPARENT;

        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:

            return analyze_glossy_bsdf(bsdf);

        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            return IMaterial_instance::OPACITY_OPAQUE;

        default:
            MDL_ASSERT(!"unhandled BSDF");
            return IMaterial_instance::OPACITY_UNKNOWN;
        }
    }

    /// Analyze if the given BSDF is opaque or transparent.
    ///
    /// \param node  a DAG expression representing the BSDF
    ///
    /// \returns opaque      if the material instance is opaque for sure
    ///          transparent if the material instance is transparent for sure
    ///          unknown     otherwise (might depend on parameters)
    Result analyze_bsdf(DAG_node const *node)
    {
        node = skip_temp(node);
        if (is<DAG_constant>(node)) {
            // invalid ref, this IS opaque
            return IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *bsdf = as<DAG_call>(node);
        if (bsdf == NULL)
            return IMaterial_instance::OPACITY_UNKNOWN;

        IDefinition::Semantics sema = bsdf->get_semantic();

        if (semantic_is_operator(sema) && semantic_to_operator(sema) == IExpression::OK_TERNARY) {
            Result t_res = analyze_bsdf(bsdf->get_argument(1));
            Result f_res = analyze_bsdf(bsdf->get_argument(2));

            if (t_res == f_res)
                return t_res;
            return IMaterial_instance::OPACITY_UNKNOWN;
        }

        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_UNBOUNDED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX:
            return analyze_bsdf_mixer(bsdf);

        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            return analyze_bsdf_layerer(bsdf);


        case IDefinition::DS_INTRINSIC_DF_TINT:
        case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
        case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_COAT_ABSORPTION_FACTOR:
            return analyze_bsdf_modifier(bsdf);

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFLAKE_SHEEN_BSDF:

            return analyze_elemental_bsdf(bsdf);

        default:
            MDL_ASSERT(!"unhandled BSDF");
            return IMaterial_instance::OPACITY_UNKNOWN;
        }
    }

    /// Get a value from a constant by an absolute path.
    ///
    /// \param value  a value
    /// \param path   the path
    IValue const *get_value(IValue const *value, Array_ref<char const *> const &path)
    {
        for (size_t i = 0, n = path.size(); i < n; ++i) {
            IValue_struct const *s_value = cast<IValue_struct>(value);
            value = s_value->get_value(path[i]);
        }
        return value;
    }

    /// Get a value from an expression by absolute path.
    ///
    /// \param expr  a DAG expression
    /// \param path  the path
    IValue const *get_value(DAG_node const *expr, Array_ref<char const *> const &path)
    {
        for (size_t i = 0, n = path.size(); i < n; ++i) {
            expr = skip_temp(expr);

            if (DAG_constant const *c = as<DAG_constant>(expr)) {
                IValue const *v = c->get_value();

                return get_value(v, path.slice(i));
            }
            if (DAG_call const *call = as<DAG_call>(expr)) {
                expr = call->get_argument(path[i]);
                if (expr == NULL) {
                    MDL_ASSERT(!"wrong access path");
                    return NULL;
                }
            } else {
                // parameter, unknown
                return NULL;
            }
        }

        if (DAG_constant const *c = as<DAG_constant>(expr)) {
            return c->get_value();
        } else {
            // not a constant, cannot decide
            return NULL;
        }
    }

    /// Get a value from the instance by an absolute path.
    ///
    /// \param path  the path
    IValue const *get_value(Array_ref<char const *> const &path)
    {
        return get_value(m_constructor, path);
    }

private:
    /// The material instance construction.
    DAG_call const *m_constructor;
};

}  // anonymous

// Returns the opacity of this instance.
IMaterial_instance::Opacity
Generated_code_dag::Material_instance::get_opacity() const
{
    if (m_properties & IP_TARGET_MATERIAL_MODEL) {
        // currently we do not support opacity analysis in target material model mode
        return OPACITY_UNKNOWN;
    }

    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(expr).analyze(/*skip_cutout=*/false);
}

/// Returns the opacity of this instance.
IMaterial_instance::Opacity
Generated_code_dag::Material_instance::get_surface_opacity() const
{
    if (m_properties & IP_TARGET_MATERIAL_MODEL) {
        // currently we do not support opacity analysis in target material model mode
        return OPACITY_UNKNOWN;
    }

    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(expr).analyze(/*skip_cutout=*/true);
}

// Returns the cutout opacity of this instance if it is constant.
IValue_float const *Generated_code_dag::Material_instance::get_cutout_opacity() const
{
    if (m_properties & IP_TARGET_MATERIAL_MODEL) {
        // currently we do not support opacity analysis in target material model mode
        return NULL;
    }

    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(expr).get_cutout_opacity();
}

}  // mdl
}  // mi

