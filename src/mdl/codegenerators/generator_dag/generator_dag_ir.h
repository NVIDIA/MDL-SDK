/******************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_IR_H
#define MDL_GENERATOR_DAG_IR_H

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_generated_dag.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class Call_impl;
class IMDL;
class IType_df;
class Value_factory;
class IValue_matrix;

/// The node factory for DAG IR nodes.
class DAG_node_factory_impl : public IGenerated_code_dag::DAG_node_factory
{

public:
    /// The type of vectors of values.
    typedef vector<IValue const *>::Type Value_vector;


    /// Constructor.
    ///
    /// \param mdl             The MDL compiler interface.
    /// \param arena           The memory arena.
    /// \param value_factory   The value factory used to create values.
    /// \param internal_space  The internal space for which to compile.
    DAG_node_factory_impl(
        IMDL          *mdl,
        Memory_arena  &arena,
        Value_factory &value_factory,
        char const    *internal_space);

    /// Create a constant.
    /// \param value       The value of the constant.
    /// \returns           The created constant.
    DAG_constant const *create_constant(IValue const *value) MDL_FINAL;

    /// Create a temporary reference.
    /// \param node         The DAG IR node that is "named" by this temporary.
    /// \param index        The index of the temporary.
    /// \returns            The created temporary.
    DAG_temporary const *create_temporary(
        DAG_node const *node,
        int            index) MDL_FINAL;

    /// Create a call.
    /// \param name            The name of the called function.
    /// \param sema            The semantics of the called function.
    /// \param call_args       The call arguments of the called function.
    /// \param num_call_args   The number of call arguments.
    /// \param ret_type        The return type of the function.
    /// \returns               The created call or an equivalent IR node.
    DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type) MDL_FINAL;

    /// Create a parameter reference.
    /// \param type        The type of the parameter
    /// \param index       The index of the parameter.
    /// \returns           The created parameter.
    DAG_parameter const *create_parameter(IType const *type, int index) MDL_FINAL;

    /// Get the type factory associated with this node factory.
    /// \returns            The type factory.
    Type_factory *get_type_factory() MDL_FINAL;

    /// Get the value factory associated with this node factory.
    /// \returns            The value factory.
    Value_factory *get_value_factory() MDL_FINAL;

    /// Get the type factory associated with this node factory.
    /// \returns            The type factory.
    Type_factory const &get_type_factory() const { return *m_value_factory.get_type_factory(); }

    /// Get the value factory associated with this node factory.
    /// \returns            The value factory.
    Value_factory const &get_value_factory() const { return m_value_factory; }

    /// Clear the value table.
    void identify_clear() { m_value_table.clear(); }

    /// Check if the value table is empty.
    bool identify_empty() const { return m_value_table.empty(); }

    /// Enable common subexpression elimination.
    ///
    /// \param flag  If true, CSE will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_cse(bool flag) { bool res = m_cse_enabled; m_cse_enabled = flag; return res; }

    /// Enable optimization.
    ///
    /// \param flag  If true, CSE will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_opt(bool flag) { bool res = m_opt_enabled; m_opt_enabled = flag; return res; }

    /// Enable ignoring no-inline annotations of functions.
    ///
    /// \param flag  If true, no-inline annotation ignoring will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_ignore_noinline(bool flag) {
        bool res = m_noinline_ignored; m_noinline_ignored = flag; return res;
    }

    /// Check if inlining is allowed.
    bool is_inline_allowed() const { return m_inline_allowed; }

    /// Check if no-inline should be ignored.
    bool is_noinline_ignored() const { return m_noinline_ignored; }

    /// Check if the state module must be imported.
    bool needs_state_import() const { return m_needs_state_import; }

    /// Check if the vidia::df module must be imported.
    bool needs_nvidia_df_import() const { return m_needs_nvidia_df_import; }

    /// Mark that nvidia::df must be imported.
    void import_nvidia_df() { m_needs_nvidia_df_import = true; }

    /// Enable function call inlining.
    ///
    /// \param flag  If true, inlining will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_inline(bool flag) {
        bool res = m_inline_allowed; m_inline_allowed = flag; return res;
    }

    /// Create a float4x4 identity matrix.
    ///
    /// \param value_factory       The value factory to use for constructing the result.
    /// \returns                   The identity matrix.
    static IValue_matrix const *create_identity_matrix(IValue_factory &value_factory);

    /// Normalize the arguments of a binary expression for better CSE support.
    ///
    /// \param op  The opcode of the binary expression.
    /// \param l   The left argument.
    /// \param r   The right argument.
    ///
    /// \return true if the arguments were exchanged, false otherwise.
    static bool normalize(
        IExpression_binary::Operator &op,
        DAG_node const               *&l,
        DAG_node const               *&r);

    /// Attempt to apply a unary operator to a value.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param op               The operator to apply.
    /// \param value            The argument value to apply the operator to.
    /// \returns                The computed value,
    ///                         or NULL if the operator does not apply to the value.
    static IValue const *apply_unary_op(
        IValue_factory                    &value_factory,
        IExpression_unary::Operator const op,
        IValue const                      *value);

    /// Attempt to apply a binary operator to two values.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param op               The operator to apply.
    /// \param left             The left argument value to apply the operator to.
    /// \param right            The right argument value to apply the operator to.
    /// \returns                The computed value,
    ///                         or null if the operator does not apply to the values.
    ///
    static IValue const *apply_binary_op(
        IValue_factory                     &value_factory,
        IExpression_binary::Operator const op,
        IValue const                       *left,
        IValue const                       *right);

    /// Convert a value to a value of type target_type.
    ///
    /// \param value_factory    The value factory to use for constructing the result.
    /// \param target_type      The target type.
    /// \param value            The value to convert.
    /// \returns                The converted value, or NULL if no conversion is possible.
    static IValue const *convert(
        IValue_factory &value_factory,
        IType const    *target_type,
        IValue const   *value);

    /// Evaluate a constructor call.
    ///
    /// \param value_factory       The value factory to use for constructing the result.
    /// \param sema                The semantic of the constructor.
    /// \param ret_type            The return type of the constructor.
    /// \param arguments           The arguments of the constructor call.
    /// \returns                   The value created by the constructor or NULL.
    static IValue const *evaluate_constructor(
        IValue_factory         &value_factory,
        IDefinition::Semantics sema,
        IType const            *ret_type,
        Value_vector const     &arguments);

    /// Evaluate an intrinsic function call.
    ///
    /// \param sema                The semantic of the intrinsic function.
    /// \param arguments           The arguments of the intrinsic function call.
    /// \param n_args              The number of arguments
    /// \returns                   The value returned by function call or NULL.
    IValue const *evaluate_intrinsic_function(
        IDefinition::Semantics sema,
        IValue const * const   arguments[],
        size_t                 n_args) const;

    /// Set a tag, version pair into a resource literal.
    ///
    /// \param lit  the literal
    /// \param tag  the tag value
    /// \param ver  the tag version
    void set_resource_tag(DAG_constant const *c, int tag, unsigned ver);

    /// Retrieve the current call evaluator.
    ICall_evaluator *get_call_evaluator() const { return m_call_evaluator; }

    /// Set the current call evaluator.
    ///
    /// \param evaluator  the new call evaluator
    void set_call_evaluator(ICall_evaluator *evaluator) { m_call_evaluator = evaluator; }

    /// Enable the folding of scene unit conversion functions.
    ///
    /// \param mdl_meters_per_scene_unit  The value for the meter/scene unit conversion.
    void enable_unit_conv_fold(float mdl_meters_per_scene_unit);

    /// Enable the folding of state::wavelength_[min|max] functions.
    ///
    /// \param wavelength_min  The value for state::wvaelength_min().
    /// \param wavelength_max  The value for state::wvaelength_max().
    void enable_wavelength_fold(
        float wavelength_min,
        float wavelength_max);


    /// Get the internal space.
    char const *get_internal_space() const { return m_internal_space; }

    /// Check if this node factory owns the given DAG node.
    bool is_owner(DAG_node const *n) const;

private:
    /// Build a call to a conversion from a ::tex::gamma value to int.
    ///
    /// \param x   the value to convert
    DAG_node const *build_gamma_conv(
        DAG_node const *x);

    /// Build a call to a operator== for a ::tex::gamma value.
    ///
    /// \param x   the left gamma value
    /// \param y   the right gamma value
    DAG_node const *build_gamma_equal(
        DAG_node const *x,
        DAG_node const *y);

    /// Build a call to a ternary operator for a texture.
    ///
    /// \param cond       the condition
    /// \param true_tex   the texture returned in the true case
    /// \param false_tex  the texture returned in the false case
    DAG_node const *build_texture_ternary(
        DAG_node const *cond,
        DAG_node const *true_tex,
        DAG_node const *false_tex);

    /// Avoid non-const gamma textures.
    ///
    /// \param tex_type  the type of the texture
    /// \param url       the resource url
    /// \param gamma     the non-const gamma expression
    DAG_node const *do_avoid_non_const_gamma(
        IType_texture const *tex_type,
        DAG_constant const  *url,
        DAG_node const      *gamma);

    /// Create an operator call.
    /// \param name            The name of the called operator.
    /// \param op              The operator.
    /// \param call_args       The arguments of the called operator.
    /// \param ret_type        The return type of the function.
    /// \returns               The created call or an equivalent IR node.
    DAG_node const *create_operator_call(
        char const                    *name,
        IExpression::Operator         op,
        DAG_call::Call_argument const call_args[],
        IType const                   *ret_type);

    /// Create a constructor call.
    /// \param name            The name of the called constructor.
    /// \param sema            The semantics of the called constructor.
    /// \param call_args       The arguments of the called constructor.
    /// \param num_call_args   The number of call arguments.
    /// \param ret_type        The return type of the constructor.
    /// \returns               The created call or an equivalent IR node.
    DAG_node const *create_constructor_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type);

    /// Create a df::*_mix() call.
    /// \param name            The name of the called function.
    /// \param sema            The semantics of the called function.
    /// \param call_arg        The call argument of the called function.
    /// \param param_name      The name of the call parameter.
    /// \param ret_type        The return type of the function.
    /// \returns               The created call or an equivalent IR node.
    DAG_node const *create_mix_call(
        char const             *name,
        IDefinition::Semantics sema,
        DAG_node const         *call_arg,
        char const             *param_name,
        IType const            *ret_type);

    /// Creates an invalid reference (i.e. a call to the a default df constructor).
    ///
    /// \param df_type  The df type of the constructor.
    DAG_node const *create_default_df_constructor(IType_df const *df_type);

    /// Remove zero weigthed components for df::normalized_mix() and color_normalized_mix().
    ///
    /// \param components       The components argument of df::[color_]normalized_mix.
    /// \param is_final_result  If true, the components array was reduced to one df.
    ///
    /// \returns A filtered IR node or NULL if the components set is empty.
    DAG_node const *remove_zero_components(
        DAG_node const *components,
        bool           &is_final_result);

    /// Remove clamped components for df::clamped_mix().
    ///
    /// \param components  The components argument of df::clamped_mix() and df::color_clamped_mix().
    /// \param is_final_result  If true, the components array was reduced to one df.
    ///
    /// \returns A filtered IR node.
    DAG_node const *remove_clamped_components(
        DAG_node const *components,
        bool           &is_final_result);

    /// Returns node or an identical IR node.
    DAG_node *identify_remember(DAG_node *node);

    /// Get the allocator of this factory.
    IAllocator *get_allocator() const { return m_builder.get_arena()->get_allocator(); }

    /// Get the field name from a getter function call name.
    string get_field_name(char const *call_name) const;

    /// Allocate a Call node.
    Call_impl *alloc_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        size_t                        num_call_args,
        IType const                   *ret_type);

    /// Unwrap a float to color cast, i.e. return a node computing a float from
    /// a node computing a color.
    DAG_node const *unwrap_float_to_color(DAG_node const *n);

private:
    /// The arena builder.
    Arena_builder m_builder;

    /// The mdl interface.
    mi::base::Handle<IMDL> m_mdl;

    /// The value factory.
    Value_factory &m_value_factory;

    /// The internal space for which to compile.
    char const *m_internal_space;

    /// The call evaluator to be used if any.
    ICall_evaluator *m_call_evaluator;

    /// The next unique ID that will be assigned.
    size_t m_next_id;

    /// If set, CSE is enabled.
    bool m_cse_enabled;

    /// If set, optimizations are enabled.
    bool m_opt_enabled;

    /// If set, inlining is allowed.
    bool m_inline_allowed;

    /// If set, no-inline annotations will be ignored.
    bool m_noinline_ignored;

    /// If set, the state module must be imported due to some transformations.
    bool m_needs_state_import;

    /// If set, the nvidia::df module must be imported due to some transformations.
    bool m_needs_nvidia_df_import;

    /// If set, texture constructors with non-const gamma argument are avoided.
    bool m_avoid_non_const_gamma;

    /// If set, scene unit conversion functions are folded.
    bool m_enable_scene_conv_fold;

    /// If set, state::wavelenth_[min|max]() functions are folded.
    bool m_enable_wavelength_fold;


    /// The meter/scene unit conversion value.
    float m_mdl_meters_per_scene_unit;

    /// The value of state::wavelength_min().
    float m_state_wavelength_min;

    /// The value of state::wavelength_max().
    float m_state_wavelength_max;

    /// A hash functor for DAG IR nodes.
    struct Hash_dag_node {
        size_t operator()(DAG_node const *node) const;
    };

    /// An Equal functor for DAG IR nodes.
    struct Equal_dag_node {
        bool operator()(DAG_node const *a, DAG_node const *b) const;
    };

    typedef hash_set<
        DAG_node *,
        Hash_dag_node,
        Equal_dag_node
    >::Type Value_table;

    /// The Value table for common subexpression elimination.
    Value_table m_value_table;
};

/// RAII scope "NO-CSE"
class No_CSE_scope {
public:
    /// RAII constructor.
    ///
    /// \param factory  the node factory on which CSE will be switched off
    explicit No_CSE_scope(DAG_node_factory_impl &factory)
    : m_factory(factory)
    , m_flag(factory.enable_cse(false))
    {
    }

    /// RAII destructor.
    ~No_CSE_scope() { m_factory.enable_cse(m_flag); }

private:
    /// the factory to switch
    DAG_node_factory_impl &m_factory;
    /// the old CSE flag
    bool m_flag;
};

/// RAII scope "NO-inline"
class No_INLINE_scope {
public:
    /// RAII constructor.
    ///
    /// \param factory  the node factory on which CSE will be switched off
    explicit No_INLINE_scope(DAG_node_factory_impl &factory)
    : m_factory(factory)
    , m_flag(factory.enable_inline(false))
    {
    }

    /// RAII destructor.
    ~No_INLINE_scope() { m_factory.enable_inline(m_flag); }

private:
    /// the factory to switch
    DAG_node_factory_impl &m_factory;
    /// the old INLINE flag
    bool m_flag;
};

/// RAII scope "inline"
class INLINE_scope {
public:
    /// RAII constructor.
    ///
    /// \param factory  the node factory on which CSE will be switched on
    explicit INLINE_scope(DAG_node_factory_impl &factory)
    : m_factory(factory)
    , m_flag(factory.enable_inline(true))
    {
    }

    /// RAII destructor.
    ~INLINE_scope() { m_factory.enable_inline(m_flag); }

private:
    /// the factory to switch
    DAG_node_factory_impl &m_factory;
    /// the old INLINE flag
    bool m_flag;
};

/// RAII scope "NO.optimization"
class No_OPT_scope {
public:
    /// RAII constructor.
    ///
    /// \param factory  the node factory on which OPTIOMIZATION will be switched off
    explicit No_OPT_scope(DAG_node_factory_impl &factory)
    : m_factory(factory)
    , m_flag(factory.enable_opt(false))
    {
    }

    /// RAII destructor.
    ~No_OPT_scope() { m_factory.enable_opt(m_flag); }

private:
    /// the factory to switch
    DAG_node_factory_impl &m_factory;
    /// the old OPT flag
    bool m_flag;
};

/// Set the index of an parameter.
void set_parameter_index(DAG_parameter *param, Uint32 param_idx);

} // mdl
} // mi

#endif

