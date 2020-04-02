/******************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "generator_dag_generated_dag.h"
#include "generator_dag_tools.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

namespace mi {
namespace mdl {

namespace {

/// Helper class to check the opacity of a material instance.
class Opacity_analyzer {
    // Must be kept in sync with ::df module
    enum scatter_mode {
        scatter_reflect,
        scatter_transmit,
        scatter_reflect_transmit
    };

public:
    typedef IGenerated_code_dag::IMaterial_instance::Opacity Result;

    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param material  the material instance construction
    Opacity_analyzer(
        IAllocator     *alloc,
        DAG_call const *material)
    : m_alloc(alloc)
    , m_constructor(material)
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
        if (skip_cutout == false) {
            IValue_float const *f_value = get_cutout_opacity();
            if (f_value == NULL) {
                // cannot analyze
                return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
            }
            if (f_value->get_value() < 1.0f) {
                // not opaque
                return IGenerated_code_dag::IMaterial_instance::OPACITY_TRANSPARENT;
            }
        }
        // We do not allow different transmission of front and back-side of an MDL material.
        // Hence it is enough to analyze the front-side.
        DAG_node const *frontside = skip_temp(m_constructor->get_argument("surface"));
        if (is<DAG_constant>(frontside)) {
            // only ONE invalid BSDF, this IS opaque
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *fs = as<DAG_call>(frontside);
        if (fs == NULL) {
            // a parameter, cannot decide
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }

        DAG_node const *scattering = skip_temp(fs->get_argument("scattering"));
        if (is<DAG_constant>(scattering)) {
            // only ONE invalid BSDF, this IS opaque
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *sc = as<DAG_call>(scattering);
        if (sc == NULL) {
            // a parameter, cannot decide
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
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
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *arr = as<DAG_call>(components);
        if (arr == NULL) {
            // a parameter, cannot decide
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }
        if (arr->get_semantic() != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
            // not an array constructor, unsupported
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }

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
                return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
            }
            if (elem_const->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
                // not a struct constructor, cannot decide
                return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
            }

            DAG_node const *bsdf = elem_const->get_argument("component");
            Result res = analyze_bsdf(bsdf);
            if (res != IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE)
                return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }

        // all good
        return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
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
        DAG_node const *lower_layer = skip_temp(bsdf->get_argument("base"));
        DAG_node const *upper_layer = skip_temp(bsdf->get_argument("layer"));

        Result low_res = analyze_bsdf(lower_layer);
        Result up_res  = analyze_bsdf(upper_layer);

        if (low_res == up_res)
            return low_res;
        return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
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
        int refl_type = scatter_reflect;
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
            break;

        default:
            MDL_ASSERT(!"unhandled glossy BSDF");
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }

        if (has_mode) {
            IValue const *v = get_value(bsdf, "mode");
            if (v == NULL) {
                return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
            }
            IValue_int_valued const *i_v = cast<IValue_int_valued>(v);
            refl_type = static_cast<scatter_mode>(i_v->get_value());
        }

        if (refl_type == scatter_transmit || refl_type == scatter_reflect_transmit) {
            return IGenerated_code_dag::IMaterial_instance::OPACITY_TRANSPARENT;
        }
        return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
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
            // MaterialLayerBSDF_DiffuseRefl
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
            // MaterialLayerBSDF_DiffuseTrans;
            return IGenerated_code_dag::IMaterial_instance::OPACITY_TRANSPARENT;

        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:

            return analyze_glossy_bsdf(bsdf);

        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;

        default:
            MDL_ASSERT(!"unhandled BSDF");
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
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
            return IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE;
        }
        DAG_call const *bsdf = as<DAG_call>(node);
        if (bsdf == NULL)
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;

        IDefinition::Semantics sema = bsdf->get_semantic();

        if (semantic_is_operator(sema) && semantic_to_operator(sema) == IExpression::OK_TERNARY) {
            Result t_res = analyze_bsdf(bsdf->get_argument(1));
            Result f_res = analyze_bsdf(bsdf->get_argument(2));

            if (t_res == f_res)
                return t_res;
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
        }

        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
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
            return analyze_bsdf_modifier(bsdf);

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
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

            return analyze_elemental_bsdf(bsdf);

        default:
            MDL_ASSERT(!"unhandled BSDF");
            return IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN;
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
    /// The allocator.
    IAllocator     *m_alloc;

    /// The material instance construction.
    DAG_call const *m_constructor;
};

}  // anonymous

// Returns the opacity of this instance.
IGenerated_code_dag::IMaterial_instance::Opacity
Generated_code_dag::Material_instance::get_opacity() const
{
    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(get_allocator(), expr).analyze(/*skip_cutout=*/false);
}

/// Returns the opacity of this instance.
IGenerated_code_dag::IMaterial_instance::Opacity
Generated_code_dag::Material_instance::get_surface_opacity() const
{
    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(get_allocator(), expr).analyze(/*skip_cutout=*/true);
}

// Returns the cutout opacity of this instance if it is constant.
IValue_float const *Generated_code_dag::Material_instance::get_cutout_opacity() const
{
    DAG_call const *expr = get_constructor();

    return Opacity_analyzer(get_allocator(), expr).get_cutout_opacity();
}

}  // mdl
}  // mi

