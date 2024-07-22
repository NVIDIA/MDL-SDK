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

#ifndef MDL_GENERATOR_DAG_DESTILLER_PLUGIN_API_IMPL_H
#define MDL_GENERATOR_DAG_DESTILLER_PLUGIN_API_IMPL_H

#include <mi/mdl/mdl_distiller_plugin_api.h>
#include <mi/mdl/mdl_distiller_rules.h>

#include <mi/mdl/mdl_generated_dag.h>

#include "generator_dag_generated_dag.h"
#include "generator_dag_ir_checker.h"

namespace mi {
namespace mdl {

class ICall_name_resolver;

///
/// The rule engine handles the transformation of a compiled material by a rule set.
///
class Distiller_plugin_api_impl :
    public IDistiller_plugin_api,
    private IGenerated_code_dag::DAG_node_factory
{
    typedef IDistiller_plugin_api Base;
public:
    typedef ptr_hash_map<DAG_node const, DAG_node const *>::Type Visited_node_map;

    /// Constructor.
    ///
    /// \param alloc          the allocator
    /// \param inst           a material instance used to retrieve an allocator
    /// \param call_resolver  a MDL call name resolver for the IR checker
    Distiller_plugin_api_impl(
        IAllocator               *alloc,
        IMaterial_instance const *instance,
        ICall_name_resolver      *call_resolver);

    virtual ~Distiller_plugin_api_impl() {}

    virtual void debug_node(IOutput_stream *outs, DAG_node const *node);

    void dump_attributes(IMaterial_instance const *inst);
    void dump_attributes(IMaterial_instance const *inst,
                         DAG_node const *node);

    void dump_attributes(IMaterial_instance const *inst, std::ostream &outs);
    void dump_attributes(IMaterial_instance const *inst,
                         DAG_node const *node, std::ostream &outs);

    void set_attribute(DAG_node const * node, char const *name,
                       DAG_node const *value) MDL_FINAL;
    void set_attribute(IMaterial_instance const *inst,
                       DAG_node const * node, char const *name,
                       mi::Float32 value) MDL_FINAL;
    void set_attribute(IMaterial_instance const *inst,
                       DAG_node const * node, char const *name,
                       mi::Sint32 value) MDL_FINAL;
    void remove_attributes(DAG_node const * node) MDL_FINAL;
    DAG_node const * get_attribute(DAG_node const * node, char const *name) MDL_FINAL;
    bool attribute_exists(DAG_node const * node, char const *name) MDL_FINAL;
    void move_attributes(DAG_node const *to_node, DAG_node const *from_node) MDL_FINAL;

    /// Apply rules using a strategy.
    ///
    /// \param inst           a compiled material instance
    /// \param matcher        a rule set matcher
    /// \param event_handler  if non-NULL, a event handler to report events during processing
    /// \param strategy       the strategy to use
    ///
    /// \return a new compiled material
    IMaterial_instance *apply_rules(
        IMaterial_instance const         *inst,
        IRule_matcher                    &matcher,
        IRule_matcher_event              *event_handler,
        const mi::mdl::Distiller_options *options,
        mi::Sint32                       &error)  MDL_FINAL;

    /// Returns a new material instance as a merge of two material instances based
    /// on a material field selection mask choosing the top-level material fields
    /// between the two materials.
    ///
    /// \param m0    the material instance whose fields are chosen if the mask bit is 0.
    /// \param m1    the material instance whose fields are chosen if the mask bit is 1.
    /// \param field_selector    mask to select the fields from m0 or m1 respectively.
    ///
    /// \return a new compiled material instance.
    IMaterial_instance *merge_materials(
        IMaterial_instance const              *m0,
        IMaterial_instance const              *m1,
        IDistiller_plugin_api::Field_selector field_selector)  MDL_FINAL;

    /// Create a constant.
    ///
    /// \param  value       The value of the constant.
    ///
    /// \returns            The created constant.
    DAG_constant const *create_constant(
        IValue const *value)  MDL_FINAL;

    /// Create a temporary reference.
    ///
    /// \param node         The DAG node that is "named" by this temporary.
    /// \param index        The index of the temporary.
    ///
    /// \returns            The created temporary reference.
    DAG_temporary const *create_temporary(DAG_node const *node, int index)  MDL_FINAL;

    /// Create a call.
    ///
    /// \param  name            The name of the called function.
    /// \param  sema            The semantics of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the function.
    ///
    /// \returns                The created call or an equivalent expression.
    DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type)  MDL_FINAL;

    /// Create a function call for a non-overloaded function. All parameter
    /// and return types are deduced from the function definition.
    ///
    /// \param  name            The name of the called function, e.g., "::state::normal".
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    ///
    /// \returns                The created call or an equivalent expression.
    DAG_node const *create_function_call(
        char const             *name,
        DAG_node const * const call_args[],
        size_t                 num_call_args)  MDL_FINAL;

    /// Create a 1-, 2-, or 3-mixer call, with 2, 4, or 6 parameters respectively.
    DAG_node const *create_mixer_call(
        DAG_call::Call_argument const call_args[],
        int                           num_call_args)  MDL_FINAL;

    /// Create a 1-, 2-, or 3-color-mixer call, with 2, 4, or 6 parameters respectively.
    DAG_node const *create_color_mixer_call(
        DAG_call::Call_argument const call_args[],
        int                           num_call_args)  MDL_FINAL;

    /// Create a parameter reference.
    ///
    /// \param  type        The type of the parameter
    /// \param  index       The index of the parameter.
    ///
    /// \returns            The created parameter reference.
    DAG_parameter const *create_parameter(IType const *type, int index)  MDL_FINAL;

    /// Enable common subexpression elimination.
    ///
    /// \param flag  If true, CSE will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_cse(bool flag) MDL_FINAL;

    /// Enable optimization.
    ///
    /// \param flag  If true, optimizations in general will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_opt(bool flag) MDL_FINAL;

    /// Enable unsafe math optimizations.
    ///
    /// \param flag  If true, unsafe math optimizations will be enabled, else disabled.
    /// \return      The old value of the flag.
    bool enable_unsafe_math_opt(bool flag) MDL_FINAL;

    /// Return unsafe math optimization setting.
     ///
    /// \return      The value of the flag.
    bool get_unsafe_math_opt() const MDL_FINAL;

    /// Get the type factory associated with this expression factory.
    ///
    /// \returns            The type factory.
    IType_factory *get_type_factory()  MDL_FINAL;

    /// Get the value factory associated with this expression factory.
    ///
    /// \returns            The value factory.
    IValue_factory *get_value_factory()  MDL_FINAL;

    /// Return the type for ::df::bsdf_component
    IType const *get_bsdf_component_type()  MDL_FINAL;

    /// Return the type for ::df::edf_component
    IType const *get_edf_component_type()  MDL_FINAL;

    /// Return the type for ::df::vdf_component
    IType const *get_vdf_component_type()  MDL_FINAL;

    /// Return the type for ::df::bsdf_component
    IType const *get_bsdf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for ::df::edf_component
    IType const *get_edf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for ::df::vdf_component
    IType const *get_vdf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for ::df::color_bsdf_component
    IType const *get_color_bsdf_component_type()  MDL_FINAL;

    /// Return the type for ::df::color_edf_component
    IType const *get_color_edf_component_type()  MDL_FINAL;

    /// Return the type for ::df::color_vdf_component
    IType const *get_color_vdf_component_type()  MDL_FINAL;

    /// Return the type for ::df::color_bsdf_component
    IType const *get_color_bsdf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for ::df::color_edf_component
    IType const *get_color_edf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for ::df::color_vdf_component
    IType const *get_color_vdf_component_array_type( int n_values)  MDL_FINAL;

    /// Return the type for bool
    IType const *get_bool_type()  MDL_FINAL;

    /// Creates an operator, handles types.
    ///
    /// \param op        the operator
    /// \param o         the operand
    ///
    /// \returns a DAG representing l op r
    DAG_node const *create_unary(
        Unary_operator op,
        DAG_node const  *o)  MDL_FINAL;

    /// Creates an operator, handles types.
    ///
    /// \param op        the operator
    /// \param l         the left operand
    /// \param r         the right operand
    ///
    /// \returns a DAG representing l op r
    DAG_node const *create_binary(
        Binary_operator op,
        DAG_node const  *l,
        DAG_node const  *r)  MDL_FINAL;

    /// Creates a ternary operator.
    ///
    /// \param cond      the condition
    /// \param t_expr    the true expression
    /// \param f_expr    the false expression
    ///
    /// \returns a DAG representing cond ? t_expr : f_expr
    DAG_node const *create_ternary(
        DAG_node const *cond,
        DAG_node const *t_expr,
        DAG_node const *f_expr)  MDL_FINAL;

    /// Creates a SELECT operator on a struct or vector.
    ///
    /// \param s       a node producing a struct typed result
    /// \param member  the name of the member to select
    DAG_node const *create_select(
        DAG_node const *s,
        char const     *member)  MDL_FINAL;

    /// Creates an array constructor.
    ///
    /// \param elem_type  the element type of the array, might be NULL iff n_values > 0
    /// \param values     the array elements
    /// \param n_values   number of values
    ///
    /// \note the element type cannot be derived from the values for zero-length arrays
    DAG_node const *create_array(
        IType const            *elem_type,
        DAG_node const * const values[],
        size_t                 n_values)  MDL_FINAL;

    /// Creates a boolean constant.
    DAG_constant const *create_bool_constant(bool f)  MDL_FINAL;

    /// Creates an integer constant.
    DAG_constant const *create_int_constant(int i)  MDL_FINAL;

    /// Creates a constant of the predefined intensity_mode enum.
    ///
    /// \param i  the index of the enum value
    DAG_constant const *create_emission_enum_constant(int i)  MDL_FINAL;

    /// Creates a constant of the df::scatter_mode enum.
    ///
    /// \param i  the index of the enum value
    DAG_constant const *create_scatter_enum_constant(int i)  MDL_FINAL;

    /// Creates a constant of the tex::wrap_mode enum.
    ///
    /// \param i  the index of the enum value
    DAG_constant const *create_wrap_mode_enum_constant(int i)  MDL_FINAL;

    /// Creates a floating point constant.
    DAG_constant const *create_float_constant(float f)  MDL_FINAL;

    /// Creates a float3 constant.
    DAG_constant const *create_float3_constant(float x, float y, float z)  MDL_FINAL;

    /// Creates a RGB color constant.
    DAG_constant const *create_color_constant( float r, float g, float b)  MDL_FINAL;

    /// Creates a RGB color constant of the global material IOR value.
    DAG_constant const *create_global_ior() MDL_FINAL;

    /// Creates a float constant of the global material IOR green value.
    DAG_constant const *create_global_float_ior() MDL_FINAL;

    /// Creates a string constant.
    DAG_constant const *create_string_constant(char const *s)  MDL_FINAL;

    /// Creates an invalid bsdf.
    DAG_constant const *create_bsdf_constant()  MDL_FINAL;

    /// Creates an invalid edf.
    DAG_constant const *create_edf_constant()  MDL_FINAL;

    /// Creates an invalid vdf.
    DAG_constant const *create_vdf_constant()  MDL_FINAL;

    /// Creates an invalid hair_bsdf.
    DAG_constant const *create_hair_bsdf_constant()  MDL_FINAL;

    /// Create a bsdf_component for a mixer; can be a call or a constant.
    DAG_node const *create_bsdf_component(
        DAG_node const* weight_arg,
        DAG_node const* bsdf_arg)  MDL_FINAL;

    /// Create a edf_component for a mixer; can be a call or a constant.
    DAG_node const *create_edf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg)  MDL_FINAL;

    /// Create a vdf_component for a mixer; can be a call or a constant.
    DAG_node const *create_vdf_component(
        DAG_node const* weight_arg,
        DAG_node const* vdf_arg)  MDL_FINAL;

    /// Create a bsdf_color_component for a color mixer; can be a call or a constant.
    DAG_node const *create_color_bsdf_component(
        DAG_node const* weight_arg,
        DAG_node const* bsdf_arg)  MDL_FINAL;

    /// Create a edf_color_component for a color mixer; can be a call or a constant.
    DAG_node const *create_color_edf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg)  MDL_FINAL;

    /// Create a vdf_color_component for a color mixer; can be a call or a constant.
    DAG_node const *create_color_vdf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg)  MDL_FINAL;

    /// Create a constant node for a given type and value.
    DAG_constant const* mk_constant( const char* const_type, const char* value)  MDL_FINAL;

    /// Create DAG_node's for possible default values of Node_types parameter.
    DAG_node const* mk_default( const char* param_type, const char* param_default)  MDL_FINAL;

    /// Returns the argument count if node is non-null and of the call kind or a compound constant,
    /// and 0 otherwise.
    size_t get_compound_argument_size(DAG_node const* node)  MDL_FINAL;

    /// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
    /// and NULL otherwise.
    DAG_node const *get_compound_argument(DAG_node const* node, size_t i)  MDL_FINAL;

    /// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
    /// and NULL otherwise; remaps index for special case handling of mixers and parameter
    /// order of glossy BSDFs.
    DAG_node const *get_remapped_argument(DAG_node const* node, size_t i)  MDL_FINAL;

    /// Returns the name of the i-th parameter of node, or NULL if there is none or node is NULL.
    char const *get_compound_parameter_name(DAG_node const *node, size_t i) const  MDL_FINAL;

    /// Returns true if node evaluates to true
    bool eval_if( DAG_node const* node)  MDL_FINAL;

    /// Returns true if node is not evaluating to false, i.e., it either evaluates
    /// to true or cannot be evaluated.
    bool eval_maybe_if( DAG_node const* node)  MDL_FINAL;

    /// Compute the node selector for the matcher, either the semantic for a DAG_call
    /// node, or one of the Distiller_extended_node_semantics covering DAG_constant 
    /// of type bsdf, edf or vdf respectively, or for DAG_constant's and DAG_call's of 
    /// one of the material structs, and selectors for mix_1, mix_2, mix_3, 
    /// clamped_mix_1, ..., as well as a special selector for local_normal.
    /// All other nodes return 0.
    int get_selector( DAG_node const* node) const  MDL_FINAL;

    /// Checks recursively for all call nodes if the property test_fct returns true.
    bool all_nodes(
        IRule_matcher::Checker_function test_fct,
        DAG_node const *node) MDL_FINAL;

    /// Set the normalization of mixer node flag and return its previous value.
    bool set_normalize_mixers( bool new_value)  MDL_FINAL;

    /// Normalize mixer nodes and set respective flag to keep them normalized
    IMaterial_instance *normalize_mixers(
        IMaterial_instance const         *inst,
        IRule_matcher_event              *event_handler,
        mi::mdl::Distiller_options const *options,
        mi::Sint32                       &error) MDL_FINAL;

    /// Immediately deletes this distiller plugin API
    void release() const MDL_FINAL;

private:
    /// Convert a enum typed value to int.
    DAG_node const *convert_enum_to_int(DAG_node const *);

    /// Replace standard material structures with a DAG_call if they happen to be constants.
    /// Also replaces mixers with a constant array with bsdf()
    DAG_node const *replace_constant_by_call(
        DAG_node const *node);

    /// Convert a material emission value into a constructor as a call.
    ///
    /// \param volume_value  the volume value that will to be converted
    DAG_node const *conv_material_emission_value(
        IValue_struct const *emission_value);

    /// Convert a material surface value into a constructor as a call.
    ///
    /// \param surface_value  the surface value that will to be converted
    DAG_node const *conv_material_surface_value(
        IValue_struct const *surface_value);

    /// Convert a material volume value into a constructor as a call.
    ///
    /// \param volume_value  the volume value that will to be converted
    DAG_node const *conv_material_volume_value(
        IValue_struct const *volume_value);

    /// Convert a material geometry value into a constructor as a call.
    ///
    /// \param geom_value  the geometry value that will to be converted
    DAG_node const *conv_material_geometry_value(
        IValue_struct const *geom_value);

    /// Convert a material value into a constructor as a call.
    ///
    /// \param material_value  the material value that will to be converted
    DAG_node const *conv_material_value(
        IValue_struct const *material_value);

    /// Do replacement using a strategy.
    ///
    /// \param root  the root node of a sub-DAG
    /// \param path  if options->trace is true, this contains the path to the current node
    ///
    /// \return the result node or a copy on the current DAG
    DAG_node const *replace(
        DAG_node const *root,
        string const   &path,
        Visited_node_map &marker_map);

    /// Checks recursively for all call nodes if the property test_fct returns true.
    bool all_nodes_rec(
        IRule_matcher::Checker_function test_fct,
        DAG_node const                 *node,
        char const                     *path,
        Visited_node_map &marker_map);

    /// Creates a (deep) copy of a node.
    ///
    /// \param root  the DAG root node to copy
    ///
    /// \return a copy of the DAG
    DAG_node const *copy_dag(
        DAG_node const *root,
        Visited_node_map &marker_map);

    /// Copies all attributes (deeply) from one node to another.
    ///
    /// \param to_node  Node to move attributes to
    /// \param from_node  Node to move attriubutes from. All attributes will be deleted from this node.
    void move_attributes_deep(DAG_node const *to_node, DAG_node const *from_node,
                              Visited_node_map &marker_map, int level,
                              bool ignore_mismatched_nodes);

    /// Get a standard library module.
    ///
    /// \param mod_name  the module name (with :: prefix)
    Module const *find_builtin_module(
        char const *mod_name);

    /// Find an exported definition of a module.
    ///
    /// \param mod   an MDL module
    /// \param name  the name of an exported MDL entity
    Definition *find_exported_def(
        Module const *mod,
        char const *name);

    /// Retrieve the allocator.
    IAllocator *get_allocator() const { return m_alloc; }

    void import_attributes(IMaterial_instance const *inst);

    void pprint_attributes(IMaterial_instance const *inst,
                           DAG_node const *node,
                           int level, std::ostream &outs);
    void pprint_node(IMaterial_instance const *inst,
                     DAG_node const *node,
                     int level, std::ostream &outs);
    void pprint_material(IMaterial_instance const *inst, std::ostream &outs);

private:
    /// The allocator.
    IAllocator            *m_alloc;

    /// The MDL compiler
    mi::base::Handle<IMDL> m_compiler;

    /// The name printer.
    Name_printer          m_printer;

    /// The type factory to be used.
    IType_factory         *m_type_factory;

    /// The value factory to be used.
    IValue_factory        *m_value_factory;

    /// The node factory to be used.
    DAG_node_factory_impl *m_node_factory;

    /// The marker map for walking DAGs.
//    Visited_node_map m_marker_map;

    /// The marker map for walking DAGs.
//    Visited_node_map m_attribute_marker_map;

    /// The current strategy.
    Rule_eval_strategy m_strategy;

    /// The current rule matcher.
    IRule_matcher *m_matcher;

    /// The event handler to report events if any.
    IRule_matcher_event *m_event_handler;

    /// Options to customize rules.
    mi::mdl::Distiller_options const *m_options;

    /// The call name resolver for the IR-checker.
    ICall_name_resolver     *m_call_resolver;

    /// The IR checker.
    DAG_ir_checker m_checker;

    /// Global IOR of the current material if ot is constant, and 1.4 otherwise
    float m_global_ior[3];

    /// Flag to control normalization of mixer nodes.
    bool  m_normalize_mixers;

    /// Map for holding attribute sets of a single node.
    typedef mi::mdl::map<char const *, DAG_node const *, strcmp_string_less>::Type Node_attr_map;

    /// Map for holding attribute sets of DAG nodes.
    typedef mi::mdl::ptr_map<DAG_node const, Node_attr_map>::Type Attr_map;

    Attr_map m_attribute_map;
};

IDistiller_plugin_api *create_distiller_plugin_api(
    IMaterial_instance const *instance,
    ICall_name_resolver      *call_resolver);

} // mdl
} // mi

#endif // MDL_GENERATOR_DAG_DESTILLER_PLUGIN_API_IMPL_H
