/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

namespace mi {
namespace mdl {

namespace {

/// Helper class to run additional checks on materials.
class Material_checker
{
    typedef ptr_hash_map<DAG_node const, size_t>::Type Arg_map;
public:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param material  the material expression
    Material_checker(
        IAllocator     *alloc,
        DAG_call const *material)
    : m_alloc(alloc)
    , m_expr(material)
    {
    }

    /// Check if the given material_surface is the "default surface", i.e.
    /// the bsdf and the edf are invalid
    bool is_default_surface(DAG_node const *surface)
    {
        if (is<DAG_constant>(surface)) {
            // can be only a literal, if bsdf and edf are invalid
            return true;
        }

        // otherwise must be the a constructor
        DAG_call const *call = as<DAG_call>(surface);
        {
            if (call == NULL || call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
                return false;
        }

        // scattering must be a literal
        DAG_node const *scattering = call->get_argument("scattering");
        if (!is<DAG_constant>(scattering))
            return false;

        // emission must be a call
        DAG_call const *emission = as<DAG_call>(call->get_argument("emission"));
        {
            if (emission == NULL || emission->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
                return false;
        }

        // edf must be a literal
        DAG_node const *edf = emission->get_argument("emission");
        if (!is<DAG_constant>(edf))
            return false;

        // ok
        return true;
    }

    /// Check the material.
    ///
    /// \returns true if the material passed all checks
    bool check()
    {
        // must be a material constructor
        MDL_ASSERT(m_expr->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR);

        if (!is_thin_walled()) {
            // not thin walled, material is ok
            return true;
        }

        return check_transmission();
    }

private:
    /// Get the thin_walled flag from a material.
    DAG_node const *get_thin_walled()
    {
        return m_expr->get_argument("thin_walled");
    }

    /// Check if the given material expression "is thin walled".
    bool is_thin_walled()
    {
        // get the thin_walled flag
        DAG_node const *flag = get_thin_walled();
        MDL_ASSERT(flag != NULL);

        if (DAG_constant const *c = as<DAG_constant>(flag)) {
            // a constant, good
            IValue_bool const *b = cast<IValue_bool>(c->get_value());
            return b->get_value();
        }

        // a complex expression that we could not analyze further, remember its position
        return true;
    }

    /// Get the scatter_mode parameter of a elemental BSDF.
    ///
    /// \param bsdf                the elemental bsdf expression call
    /// \param assume_bad          the "bad" return value if the expression is too complex
    bool get_scatter_mode_transmit(
        DAG_call const *bsdf,
        bool           assume_bad)
    {
        DAG_node const *arg = bsdf->get_argument("mode");
        MDL_ASSERT(arg != NULL);

        if (DAG_constant const *c = as<DAG_constant>(arg)) {
            IValue_enum const *ev = cast<IValue_enum>(c->get_value());
            return ev->get_value() > 0;
        }
        // cannot fold
        return assume_bad;
    }

    /// Check bsdf components for no transmission.
    bool check_component_no_transmission(
        DAG_node const *component)
    {
        if (is<DAG_constant>(component)) {
            // bsdf must be empty
            return true;
        }

        DAG_call const *cons_call = cast<DAG_call>(component);
        if (cons_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
            // not the constructor, cannot analyze
            return false;
        }

        MDL_ASSERT(cons_call->get_argument_count() == 2 &&
               "unexpected parameter count for component constructor");

        // check bsdf
        {
            DAG_node const *bsdf_expr = cons_call->get_argument("component");

            if (!check_no_transmission(bsdf_expr))
                return false;
        }
        return true;
    }

    /// Check that a *_mix expression has no transmission
    bool check_mixer_no_transmission(
        DAG_call const *mixer)
    {
        DAG_node const *components = mixer->get_argument("components");

        MDL_ASSERT(components != NULL);

        if (is<DAG_constant>(components)) {
            // a constant, has no transmission
            return true;
        }

        DAG_call const *arr_cons = cast<DAG_call>(components);
        IDefinition::Semantics sema = arr_cons->get_semantic();
        if (sema != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
            // cannot analyze further
            return false;
        }

        int n_args = arr_cons->get_argument_count();
        for (int i = 0; i < n_args; ++i) {
            DAG_node const *component = arr_cons->get_argument(i);
            if (!check_component_no_transmission(component))
                return false;
        }
        return true;
    }

    /// Check that a layer expressions has no transmission.
    bool check_layer_no_transmission(
        DAG_call const *mixer)
    {
        // layer must have no transmissive
        {
            DAG_node const *layer = mixer->get_argument("layer");

            MDL_ASSERT(layer != NULL);

            if (!check_no_transmission(layer))
                return false;
        }
        // base must have no transmissive
        {
            DAG_node const *base = mixer->get_argument("base");

            MDL_ASSERT(base != NULL);

            if (!check_no_transmission(base))
                return false;
        }
        return true;
    }

    /// Check modifier expressions with two bsdfs.
    bool check_modifier_no_transmission(
        DAG_call const *modifier)
    {
        DAG_node const *base = modifier->get_argument("base");
        MDL_ASSERT(base != NULL);

        // base must have no transmission
        return check_no_transmission(base);
    }

    /// Check that the given BSDF has no transmission.
    bool check_no_transmission(DAG_node const *bsdf_expr)
    {
        if (is<DAG_constant>(bsdf_expr)) {
            // must be the invalid bsdf, has no transmission
            return true;
        }
        if (!is<DAG_call>(bsdf_expr)) {
            // not a call, we cannot analyze it
            return false;
        }

        DAG_call const *bsdf = cast<DAG_call>(bsdf_expr);

        switch (bsdf->get_semantic()) {
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            // no transmission at all
            return true;

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
            // has transmission
            return false;

        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
            return !get_scatter_mode_transmit(bsdf, /*assume_bad=*/true);

        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
            return !get_scatter_mode_transmit(bsdf, /*assume_bad=*/true);

        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
            // no transmission at all
            return true;

        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            return !get_scatter_mode_transmit(bsdf, /*assume_bad=*/true);

        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
            return check_mixer_no_transmission(bsdf);

        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            return check_layer_no_transmission(bsdf);

        case IDefinition::DS_INTRINSIC_DF_TINT:
        case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
        case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
            return check_modifier_no_transmission(bsdf);

        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            return !get_scatter_mode_transmit(bsdf, /*assume_bad=*/true);

        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
            // no transmission at all
            return true;

        default:
            // for now, too complex; assume transmission
            return false;
        }
    }

    /// Get the weight of the given component is it is constant.
    IValue_float const *get_component_const_weight(
        DAG_node const *component)
    {
        if (DAG_constant const *c = as<DAG_constant>(component)) {
            IValue_struct const *sv = cast<IValue_struct>(c->get_value());
            return cast<IValue_float>(sv->get_field("weight"));
        }
        DAG_call const *cons_call = cast<DAG_call>(component);
        if (cons_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
            return NULL;
        DAG_node const *w = cons_call->get_argument("weight");
        if (DAG_constant const *c = as<DAG_constant>(w)) {
            return cast<IValue_float>(c->get_value());
        }
        return NULL;
    }

    /// Check two bsdf components for equality.
    bool check_same_component(
        DAG_node const *left,
        DAG_node const *right)
    {
        if (left == right)
            return true;

        DAG_call const *l_call = as<DAG_call>(left);
        DAG_call const *r_call = as<DAG_call>(right);

        if (l_call == NULL || r_call == NULL)
            return check_same_expression(left, right, /*only_trans=*/false);

        int n_args = l_call->get_argument_count();
        if (n_args != r_call->get_argument_count())
            return false;

        if (l_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR ||
            r_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR) {
            // at least one of them is not the constructor
            return false;
        }

        MDL_ASSERT(n_args == 2 && "unexpected parameter count for component constructor");

        // check weight
        {
            DAG_node const *left  = l_call->get_argument("weight");
            DAG_node const *right = r_call->get_argument("weight");

            if (!check_same_expression(left, right, /*only_trans=*/false))
                return false;
        }
        // check component
        {
            DAG_node const *left  = l_call->get_argument("component");
            DAG_node const *right = r_call->get_argument("component");

            if (!check_same_expression(left, right, /*only_trans=*/true))
                return false;
        }
        return true;
    }

    /// Check two given expressions for equality.
    ///
    /// \param left        the left expression
    /// \param right       the right expression
    /// \param only_trans  if true, two bsdf expressions without transmission are equal
    bool check_same_expression(
        DAG_node const *left,
        DAG_node const *right,
        bool           only_trans)
    {
        MDL_ASSERT(left != NULL && right != NULL);

        // due to skip_temporary() we can have DAG's here, so do the simple check first
        if (left == right)
            return true;

        if (only_trans && is<IType_bsdf>(left->get_type()->skip_type_alias())) {
            if (check_no_transmission(left) && check_no_transmission(right)) {
                // both have no transmission
                return true;
            }
        }

        DAG_node::Kind kind = left->get_kind();
        if (kind != right->get_kind()) {
            // different kind
            return false;
        }
        switch (kind) {
        case DAG_node::EK_CONSTANT:
            // if two constants are not CSE'd here, they are different
            MDL_ASSERT(cast<DAG_constant>(left)->get_value() !=
                   cast<DAG_constant>(right)->get_value());
            return false;

        case DAG_node::EK_TEMPORARY:
        case DAG_node::EK_PARAMETER:
            // should not happen because temporaries are not created and we are in
            // instance compilation mode
            MDL_ASSERT(!"unexpected DAG node kind");
            return true;

        case DAG_node::EK_CALL:
            {
                DAG_call const *l_call = cast<DAG_call>(left);
                DAG_call const *r_call = cast<DAG_call>(right);

                int n_args = l_call->get_argument_count();
                if (n_args != r_call->get_argument_count())
                    return false;

                IDefinition::Semantics l_sema = l_call->get_semantic();
                IDefinition::Semantics r_sema = r_call->get_semantic();
                bool l_is_arr = l_sema == IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
                bool r_is_arr = r_sema == IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR;
                if (l_is_arr != r_is_arr) {
                    // only one is an array constructor
                    return false;
                }
                if (!l_is_arr) {
                    if (l_sema != r_sema) {
                        // different functions called
                        return false;
                    } else if (only_trans) {
                        // same functions called, check for equal transmission
                        return check_equal_transmission(l_call, r_call);
                    }
                }

                // otherwise they are different, because not CSE'd
                return false;
            }
        }
        MDL_ASSERT(!"unsupported DAG node kind");
        return true;
    }

    /// Check a call parameter for equality.
    bool check_same_parameter(
        DAG_call const *lbsdf,
        DAG_call const *rbsdf,
        char const     *name)
    {
        DAG_node const *left  = lbsdf->get_argument(name);
        DAG_node const *right = rbsdf->get_argument(name);

        return check_same_expression(left, right, /*only_trans=*/false);
    }

    /// Try summing up all non-transmissives and permuting transmissives components.
    bool check_sum_and_permute_components(
        DAG_call const *l,
        DAG_call const *r)
    {
        int n_l_args = l->get_argument_count();
        int n_r_args = r->get_argument_count();

        bool can_sum_non_trans = true;

        float l_non_tranmissive_weight = 0.0f;

        Arg_map trans_args(0, Arg_map::hasher(), Arg_map::key_equal(), m_alloc);

        for (int i = 0; i < n_l_args; ++i) {
            DAG_node const *larg = l->get_argument(i);

            if (check_component_no_transmission(larg)) {
                IValue_float const *w = get_component_const_weight(larg);
                if (w == NULL) {
                    can_sum_non_trans = false;
                    break;
                }
                l_non_tranmissive_weight += w->get_value();
            } else {
                Arg_map::iterator it = trans_args.find(larg);
                if (it == trans_args.end()) {
                    // first occurrence
                    trans_args[larg] = 1;
                } else {
                    it->second += 1;
                }
            }
        }

        if (can_sum_non_trans) {
            float r_non_tranmissive_weight = 0.0f;

            for (int i = 0; i < n_r_args; ++i) {
                DAG_node const *rarg = r->get_argument(i);

                if (check_component_no_transmission(rarg)) {
                    IValue_float const *w = get_component_const_weight(rarg);
                    if (w == NULL) {
                        can_sum_non_trans = false;
                        break;
                    }
                    r_non_tranmissive_weight += w->get_value();
                } else {
                    Arg_map::iterator it = trans_args.find(rarg);
                    if (it == trans_args.end()) {
                        can_sum_non_trans = false;
                        break;
                    }
                    it->second -= 1;
                }
            }
            if (can_sum_non_trans) {
                // should we use an epsilon here?
                if (l_non_tranmissive_weight == r_non_tranmissive_weight) {
                    for (Arg_map::iterator it(trans_args.begin()), end(trans_args.end());
                        it != end;
                        ++it)
                    {
                        if (it->second != 0) {
                            // not all references taken
                            can_sum_non_trans = false;
                            break;
                        }
                    }
                    if (can_sum_non_trans)
                        return true;
                }
            }
        }
        return false;
    }

    /// Try permuting components.
    bool check_permute_component(
        DAG_call const *l,
        DAG_call const *r)
    {
        int n_l_args = l->get_argument_count();
        int n_r_args = r->get_argument_count();

        Arg_map args(0, Arg_map::hasher(), Arg_map::key_equal(), m_alloc);

        // check for permutations first
        for (int i = 0; i < n_l_args; ++i) {
            DAG_node const *larg = l->get_argument(i);

            Arg_map::iterator it = args.find(larg);
            if (it == args.end()) {
                // first occurrence
                args[larg] = 1;
            } else {
                it->second += 1;
            }
        }

        bool is_permutation = true;
        for (int i = 0; i < n_r_args; ++i) {
            DAG_node const *rarg = r->get_argument(i);

            Arg_map::iterator it = args.find(rarg);
            if (it == args.end()) {
                is_permutation = false;
                break;
            } else {
                it->second -= 1;
            }
        }
        if (is_permutation) {
            for (Arg_map::iterator it(args.begin()), end(args.end());
                it != end;
                ++it)
            {
                if (it->second != 0) {
                    // not all references taken
                    is_permutation = false;
                    break;
                }
            }
            if (is_permutation)
                return true;
        }
        return false;
    }

    /// Check normalized_mix expressions.
    bool check_normalized_mix(
        DAG_call const *left,
        DAG_call const *right)
    {
        DAG_node const *larg = left->get_argument("components");
        DAG_node const *rarg = right->get_argument("components");

        MDL_ASSERT(larg != NULL && rarg != NULL);
        DAG_call const *l = as<DAG_call>(larg);
        DAG_call const *r = as<DAG_call>(rarg);

        if (l == NULL || r == NULL) {
            // not calls, must be identical
            return check_same_expression(larg, rarg, /*only_trans=*/false);
        }

        IDefinition::Semantics lsema = l->get_semantic();
        IDefinition::Semantics rsema = r->get_semantic();

        if (lsema != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR ||
            rsema != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
        {
            // cannot analyze further
            return false;
        }

        // first check, if all weights are constants: then we can sum up non-transmissives
        // and check transmissives for permutations
        if (check_sum_and_permute_components(l, r))
            return true;

        // check if the arguments are just a permutation
        if (check_permute_component(l, r))
            return true;

        // fallback to fully strict check
        int n_args = l->get_argument_count();
        if (n_args != r->get_argument_count()) {
            return false;
        }

        for (int i = 0; i < n_args; ++i) {
            DAG_node const *larg = l->get_argument(i);
            DAG_node const *rarg = r->get_argument(i);

            if (!check_same_component(larg, rarg))
                return false;
        }
        return true;
    }

    /// Check clamped_mix expressions.
    bool check_clamped_mix(
        DAG_call const *left,
        DAG_call const *right)
    {
        DAG_node const *larg = left->get_argument("components");
        DAG_node const *rarg = right->get_argument("components");

        MDL_ASSERT(larg != NULL && rarg != NULL);
        DAG_call const *l = as<DAG_call>(larg);
        DAG_call const *r = as<DAG_call>(rarg);

        if (l == NULL || r == NULL) {
            // not calls, must be identical
            return check_same_expression(larg, rarg, /*only_trans=*/false);
        }

        IDefinition::Semantics lsema = l->get_semantic();
        IDefinition::Semantics rsema = r->get_semantic();

        if (lsema != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR ||
            rsema != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
        {
            // cannot analyze further
            return false;
        }

        // for now, be strict
        int n_args = l->get_argument_count();
        if (n_args != r->get_argument_count()) {
            return false;
        }

        for (int i = 0; i < n_args; ++i) {
            DAG_node const *larg = l->get_argument(i);
            DAG_node const *rarg = r->get_argument(i);

            if (!check_same_component(larg, rarg))
                return false;
        }
        return true;
    }

    /// Check layer expressions with two bsdfs.
    bool check_layer(
        DAG_call const *left,
        DAG_call const *right)
    {
        // layer must be equal OR not transmissive
        {
            DAG_node const *l_layer = left->get_argument("layer");
            DAG_node const *r_layer = right->get_argument("layer");

            MDL_ASSERT(l_layer != NULL && r_layer != NULL);

            if (!check_same_expression(l_layer, r_layer, /*only_trans=*/true))
                return false;
        }
        // base must be equal OR not transmissive
        {
            DAG_node const *l_base = left->get_argument("base");
            DAG_node const *r_base = right->get_argument("base");

            MDL_ASSERT(l_base != NULL && r_base != NULL);

            if (!check_same_expression(l_base, r_base, /*only_trans=*/true))
                return false;
        }
        return true;
    }

    /// Check modifier expressions with two bsdfs.
    bool check_modifier(
        DAG_call const *left,
        DAG_call const *right)
    {
        int n_args = left->get_argument_count();
        if (n_args != right->get_argument_count()) {
            // no further errors
            return true;
        }

        // first check if base has transmission
        for (int i = 0; i < n_args; ++i) {
            char const *p_name = left->get_parameter_name(i);
            if (strcmp(p_name, "base") == 0) {
                DAG_node const *l_arg = left->get_argument(i);
                bool left_no_trans = check_no_transmission(l_arg);

                DAG_node const *r_arg = right->get_argument(p_name);
                bool right_no_trans = check_no_transmission(r_arg);

                if (left_no_trans != right_no_trans) {
                    // different transmissions
                    return false;
                }
                if (left_no_trans) {
                    // both have no transmission, ignore modifier
                    return true;
                }
            }
        }

        for (int i = 0; i < n_args; ++i) {
            char const *p_name = left->get_parameter_name(i);
            bool is_base = strcmp(p_name, "base") == 0;

            DAG_node const *l_arg = left->get_argument(i);
            DAG_node const *r_arg = right->get_argument(p_name);

            MDL_ASSERT(l_arg != NULL && r_arg != NULL);

            if (!check_same_expression(l_arg, r_arg, /*only_trans=*/is_base))
                return false;
        }
        return true;
    }

    /// Check for equal transmission.
    bool check_equal_transmission(
        DAG_node const *left,
        DAG_node const *right)
    {
        DAG_call const *lbsdf = as<DAG_call>(left);
        DAG_call const *rbsdf = as<DAG_call>(right);

        if (lbsdf == NULL || rbsdf == NULL) {
            // at least one is not a call, we cannot do much here
            return check_same_expression(left, right, /*only_trans=*/true);
        }

        IDefinition::Semantics sema = lbsdf->get_semantic();
        if (sema != rbsdf->get_semantic()) {
            // different calls: both must have NO transmission
            return check_no_transmission(lbsdf) && check_no_transmission(rbsdf);
        }

        // same call: check for equal structure
        switch (sema) {
        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
        case IDefinition::DS_INTRINSIC_DF_SHEEN_BSDF:
            // no transmission at all
            return true;

        case IDefinition::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
            // has transmission, check tint parameter
            return check_same_parameter(lbsdf, rbsdf, "tint");

        case IDefinition::DS_INTRINSIC_DF_SPECULAR_BSDF:
            {
                bool l_transmit = get_scatter_mode_transmit(lbsdf, /*assume_bad=*/true);
                bool r_transmit = get_scatter_mode_transmit(rbsdf, /*assume_bad=*/true);

                if (l_transmit != r_transmit) {
                    // different
                    return false;
                }
                if (!l_transmit) {
                    // both do NOT transmit, ok
                    return true;
                }
                // has transmission, check tint parameter
                return check_same_parameter(lbsdf, rbsdf, "tint");
            }

        case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
            {
                bool l_transmit = get_scatter_mode_transmit(lbsdf, /*assume_bad=*/true);
                bool r_transmit = get_scatter_mode_transmit(rbsdf, /*assume_bad=*/true);

                if (l_transmit != r_transmit) {
                    // different
                    return false;
                }
                if (!l_transmit) {
                    // both do NOT transmit, ok
                    return true;
                }
                // check roughness_u parameter
                if (!check_same_parameter(lbsdf, rbsdf, "roughness_u"))
                    return false;
                // check roughness_v parameter
                if (!check_same_parameter(lbsdf, rbsdf, "roughness_v"))
                    return false;
                // check tint parameter
                if (!check_same_parameter(lbsdf, rbsdf, "tint"))
                    return false;
                // check tangent_u parameter
                if (!check_same_parameter(lbsdf, rbsdf, "tangent_u"))
                    return false;
                // ok
                return true;
            }

        case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
            // no transmission at all
            return true;

        case IDefinition::DS_INTRINSIC_DF_MEASURED_BSDF:
            {
                bool l_transmit = get_scatter_mode_transmit(lbsdf, /*assume_bad=*/true);
                bool r_transmit = get_scatter_mode_transmit(rbsdf, /*assume_bad=*/true);

                if (l_transmit != r_transmit) {
                    // different
                    return false;
                }
                if (!l_transmit) {
                    // both do NOT transmit, ok
                    return true;
                }
                // check measurement parameter
                if (!check_same_parameter(lbsdf, rbsdf, "measurement"))
                    return false;
                // check multiplier parameter
                if (!check_same_parameter(lbsdf, rbsdf, "multiplier"))
                    return false;
                // ok
                return true;
            }
        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
            return check_normalized_mix(lbsdf, rbsdf);

        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
            return check_clamped_mix(lbsdf, rbsdf);

        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
        case IDefinition::DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER:
            return check_layer(lbsdf, rbsdf);

        case IDefinition::DS_INTRINSIC_DF_TINT:
        case IDefinition::DS_INTRINSIC_DF_THIN_FILM:
        case IDefinition::DS_INTRINSIC_DF_DIRECTIONAL_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR:
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_FACTOR:
            return check_modifier(lbsdf, rbsdf);

        case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
            // no transmission at all
            return true;

        default:
            // for now, too complex; assume different transmission
            return false;
        }

        return false;
    }

    /// Check transmission on both sides.
    bool check_transmission()
    {
        DAG_node const *sf = m_expr->get_argument("surface");
        MDL_ASSERT(sf != NULL);

        DAG_node const *bf = m_expr->get_argument("backface");
        MDL_ASSERT(bf != NULL);

        if (is_default_surface(bf)) {
            // no further checks needed, the backface will be copied from surface
            return true;
        }

        bool sf_no_transmission = false;
        if (is<DAG_constant>(sf)) {
            // if the whole surface is a constant, scattering must be the invalid bsdf
            sf_no_transmission = true;
        } else {
            // extract the scattering parameter if possible
            if (DAG_call const *sf_call = as<DAG_call>(sf)) {
                if (sf_call->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    sf = sf_call->get_argument("scattering");
                    if (is<DAG_constant>(sf)) {
                         // scattering must be the invalid bsdf
                        sf_no_transmission = true;
                    }
                }
            }
        }

        bool bf_no_transmission = false;
        if (is<DAG_constant>(bf)) {
            // if the whole surface is a constant, scattering must be the invalid bsdf
            bf_no_transmission = true;
        } else {
            // extract the scattering parameter if possible
            if (DAG_call const *bf_call = as<DAG_call>(bf)) {
                if (bf_call->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    bf = bf_call->get_argument("scattering");
                    if (is<DAG_constant>(bf)) {
                        // scattering must be the invalid bsdf
                        bf_no_transmission = true;
                    }
                }
            }
        }

        if (sf_no_transmission && sf_no_transmission == bf_no_transmission) {
            // no transmission on both sides, ok
            return true;
        }
        if (sf_no_transmission != bf_no_transmission) {
            if (!sf_no_transmission) {
                // sf must be without transmission
                return check_no_transmission(sf);
            } else {
                return check_no_transmission(bf);
            }
        }
        return check_equal_transmission(sf, bf);
    }

private:
    /// The allocator.
    IAllocator     *m_alloc;

    /// The material expression call.
    DAG_call const *m_expr;
};

}  // anonymous

// Check that thin walled materials have the same transmission on both sides.
bool Generated_code_dag::Material_instance::check_thin_walled_material()
{
    DAG_call const *expr = get_constructor();

    return Material_checker(get_allocator(), expr).check();
}

}  // mdl
}  // mi

