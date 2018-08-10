/******************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
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
class Opacity_checker {
    // Must be kept in sync with ::df module
    enum scatter_mode {
        scatter_reflect,
        scatter_transmit,
        scatter_reflect_transmit
    };

public:
    enum Result {
        unknown,
        transparent,
        opaque
    };

    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param material  the material instance construction
    Opacity_checker(
        IAllocator     *alloc,
        DAG_call const *material)
    : m_alloc(alloc)
    , m_constructor(material)
    {
    }

    /// Check if the given instance is opaque.
    ///
    /// \returns true if the material passed all checks
    Result is_opaque()
    {
        // must be a material constructor
        MDL_ASSERT(m_constructor->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR);

        // first check the cutout_opacity
        static char const * const path[] = { "geometry", "cutout_opacity" };
        IValue const *v = get_value(path);

        if (v == NULL) {
            // cannot analyze
            return unknown;
        }
        IValue_float const *f_value = cast<IValue_float>(v);
        if (f_value->get_value() < 1.0f) {
            // not opaque
            return transparent;
        }

        // We do not allow different transmission of front and back-side of an MDl material.
        // Hence it is enough to analyze the front-side.
        DAG_node const *frontside = skip_temp(m_constructor->get_argument("surface"));
        if (is<DAG_constant>(frontside)) {
            // only ONE invalid BSDF, this IS opaque
            return opaque;
        }
        DAG_call const *fs = as<DAG_call>(frontside);
        if (fs == NULL) {
            // a parameter, cannot decide
            return unknown;
        }

        DAG_node const *scattering = skip_temp(fs->get_argument("scattering"));
        if (is<DAG_constant>(scattering)) {
            // only ONE invalid BSDF, this IS opaque
            return opaque;
        }
        DAG_call const *sc = as<DAG_call>(scattering);
        if (sc == NULL) {
            // a parameter, cannot decide
            return unknown;
        }

        return is_bsdf_opaque(sc);
    }

private:
    /// Skip a temporary.
    DAG_node const *skip_temp(DAG_node const *expr)
    {
        if (DAG_temporary const *temp = as<DAG_temporary>(expr)) {
            expr = temp->get_expr();
        }
        return expr;
    }

    /// Check if a bsdf mixer is opaque.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    Result is_bsdf_mixer_opaque(DAG_call const *bsdf)
    {
        DAG_node const *components = skip_temp(bsdf->get_argument("component"));
        if (is<DAG_constant>(components)) {
            // can contain only invalid refs
            return opaque;
        }
        DAG_call const *arr = as<DAG_call>(components);
        if (arr == NULL) {
            // a parameter, cannot decide
            return unknown;
        }
        if (arr->get_semantic() != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
            // not an array constructor, unsupported
            return unknown;
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
                return unknown;
            }
            if (elem_const->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
                // not a struct constructor, cannot decide
                return unknown;
            }

            DAG_node const *bsdf = elem_const->get_argument("component");
            Result res = is_bsdf_opaque(bsdf);
            if (res != opaque)
                return unknown;
        }

        // all good
        return opaque;
    }

    /// Check if a bsdf layerer is opaque.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    Result is_bsdf_layerer_opaque(DAG_call const *bsdf)
    {
        DAG_node const *lower_layer = skip_temp(bsdf->get_argument("base"));
        DAG_node const *upper_layer = skip_temp(bsdf->get_argument("layer"));

        Result low_res = is_bsdf_opaque(lower_layer);
        Result up_res  = is_bsdf_opaque(upper_layer);

        if (low_res == up_res)
            return low_res;
        return unknown;
    }

    /// Check if a bsdf modifier is opaque.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    Result is_bsdf_modifier_opaque(DAG_call const *bsdf)
    {
        DAG_node const *base = skip_temp(bsdf->get_argument("base"));
        return is_bsdf_opaque(base);
    }

    /// Check if a glossy bsdf is opaque.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    Result is_glossy_bsdf_opaque(DAG_call const *bsdf)
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

        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_PHONG_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF_LEGACY:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_LEGACY_MCP_GLOSSY_BSDF:

            has_mode = true;
            break;

        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:

        case IDefinition::DS_INTRINSIC_NVIDIA_DF_ASHIKHMIN_SHIRLEY_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_WARD_GM_GLOSSY_BSDF:
            break;

        default:
            MDL_ASSERT(!"unhandled glossy BSDF");
            return unknown;
        }

        if (has_mode) {
            IValue const *v = get_value(bsdf, "mode");
            if (v == NULL) {
                return unknown;
            }
            IValue_int_valued const *i_v = cast<IValue_int_valued>(v);
            refl_type = static_cast<scatter_mode>(i_v->get_value());
        }

        if (refl_type == scatter_transmit || refl_type == scatter_reflect_transmit) {
            return transparent;
        }
        return opaque;
    }

    /// Check if an elemental bsdf is opaque.
    ///
    /// \param bsdf  a DAG expression representing the BSDF
    Result is_elemental_bsdf_opaque(DAG_call const *bsdf)
    {
        switch (bsdf->get_semantic()) {
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
            // MaterialLayerBSDF_DiffuseRefl
            return opaque;

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
            // MaterialLayerBSDF_DiffuseTrans;
            return opaque;

        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:

        case IDefinition::DS_INTRINSIC_NVIDIA_DF_ASHIKHMIN_SHIRLEY_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_WARD_GM_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_PHONG_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF_LEGACY:
            return is_glossy_bsdf_opaque(bsdf);

        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            return opaque;
        default:
            MDL_ASSERT(!"unhandled BSDF");
            return unknown;
        }
    }

    /// Check if the given BSDF is opaque.
    ///
    /// \param node  a DAG expression representing the BSDF
    Result is_bsdf_opaque(DAG_node const *node)
    {
        node = skip_temp(node);
        if (is<DAG_constant>(node)) {
            // invalid ref, this IS opaque
            return opaque;
        }
        DAG_call const *bsdf = as<DAG_call>(node);
        if (bsdf == NULL)
            return unknown;

        IDefinition::Semantics sema = bsdf->get_semantic();
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
            return is_bsdf_mixer_opaque(bsdf);

        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            return is_bsdf_layerer_opaque(bsdf);

        case IDefinition::DS_INTRINSIC_DF_TINT:
        case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
        case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
            return is_bsdf_modifier_opaque(bsdf);

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

        case IDefinition::DS_INTRINSIC_NVIDIA_DF_ASHIKHMIN_SHIRLEY_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_WARD_GM_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_BECKMANN_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_GGX_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_MICROFACET_PHONG_VC_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_NVIDIA_DF_SIMPLE_GLOSSY_BSDF_LEGACY:
            return is_elemental_bsdf_opaque(bsdf);

        default:
            MDL_ASSERT(!"unhandled BSDF");
            return unknown;
        }
    }

    /// Find the field index of a struct type.
    ///
    /// \param  s_type the struct type
    /// \param  name   the field name
    ///
    /// \return the index or -1 if the index was not found
    int find_field(IType_struct const *s_tp, char const *name)
    {
        for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
            IType const *f_type;
            ISymbol const *f_sym;

            s_tp->get_field(i, f_type, f_sym);
            if (strcmp(f_sym->get_name(), name) == 0)
                return i;
        }
        return -1;
    }

    /// Get a value from a constant by an absolute path.
    ///
    /// \param value  a value
    /// \param path   the path
    IValue const *get_value(IValue const *value, Array_ref<char const *> const &path)
    {
        for (size_t i = 0, n = path.size(); i < n; ++i) {
            IValue_struct const *s_value = cast<IValue_struct>(value);
            IType_struct const  *s_type  = s_value->get_type();

            int idx = find_field(s_type, path[i]);
            if (idx < 0) {
                MDL_ASSERT(!"wrong access path");
                return NULL;
            }

            value = s_value->get_value(i);
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

// Returns true if this instance is opaque.
bool Generated_code_dag::Material_instance::is_opaque() const
{
    DAG_call const *expr = get_constructor();

    return Opacity_checker(get_allocator(), expr).is_opaque() == Opacity_checker::opaque;
}

}  // mdl
}  // mi
