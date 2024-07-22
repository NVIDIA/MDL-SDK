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

/// \file mi/mdl/mdl_distiller_plugin_api.h
/// \brief MDL distiller plugin API

#ifndef MDL_DISTILLER_PLUGIN_API_H
#define MDL_DISTILLER_PLUGIN_API_H

#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_distiller_rules.h>

namespace mi {
namespace mdl {

class ICall_name_resolver;

/// The \c mi::mdl::IDistiller_plugin_api is versioned with this number.
/// A plugin is only accepted if it is compiled against the same API version
/// than the SDK. This version needs to be incremented whenever something in
/// this API changes.
#define MI_MDL_DISTILLER_PLUGIN_API_VERSION 4

///
/// The rule engine handles the transformation of a compiled material by a rule set.
///
class IDistiller_plugin_api
{
public:
    /// Apply rules using a strategy.
    ///
    /// \param inst           a compiled material instance
    /// \param matcher        a rule set matcher
    /// \param event_handler  if non-NULL, a event handler to report events during processing
    /// \param options        the strategy to use
    /// \param error          error codes reported back to the API
    ///
    /// \return a new compiled material
    virtual IMaterial_instance *apply_rules(
        IMaterial_instance const *inst,
        IRule_matcher                                 &matcher,
        IRule_matcher_event                           *event_handler,
        const Distiller_options                       *options,
        mi::Sint32                                    &error) = 0;

    /// == Node attributes =============================================================
    ///
    /// The following types and functions allow managing of node attributes.

    struct strcmp_string_less {
        bool operator()(char const *a, char const *b) const
        {
            return strcmp(a, b) < 0;
        }
    };

    virtual void dump_attributes(IMaterial_instance const *inst) = 0;
    virtual void dump_attributes(IMaterial_instance const *inst,
                                 DAG_node const *node) = 0;

    /// Set the value of a named attribute of a node. If the node
    /// already has the attribute, its value is overwritten.
    ///
    /// \param node  node to attach attribute to (must not be NULL)
    /// \param name  name of the attribute (must not be NULL)
    /// \param value new value of the attribute (must not be NULL)
    virtual void set_attribute(DAG_node const * node, char const *name,
                               DAG_node const *value) = 0;
    virtual void set_attribute(IMaterial_instance const *inst,
                               DAG_node const * node, char const *name,
                               mi::Float32 value) = 0;
    virtual void set_attribute(IMaterial_instance const *inst,
                               DAG_node const * node, char const *name,
                               mi::Sint32 value) = 0;

    /// Remove all attributes from the given DAG node.
    ///
    /// \param node node for which to remove all attributes
    virtual void remove_attributes(DAG_node const * node) = 0;

    /// Get the value of an attribute for the given DAG node.
    ///
    /// \param node node for which to get the attribute (must not be NULL)
    /// \param name name of the attribute (must not be NULL)
    ///
    /// \return value of the attribute or NULL if the attribute does not exist.
    virtual DAG_node const * get_attribute(DAG_node const * node, char const *name) = 0;

    /// Check whether a named attribute exists for a DAG node.
    ///
    /// \param node node for which to check for attribute presence (must not be NULL)
    /// \param name name of the attribute (must not be NULL)
    ///
    /// \return true if the attribute exists, false otherwise
    virtual bool attribute_exists(DAG_node const * node, char const *name) = 0;

    /// Move all nodes from `from_node` to node `to_node`, removing
    /// them from `from_node`.
    ///
    /// \param to_node node to which to assign the attributes
    /// \param from_node node from which to move the attributes
    virtual void move_attributes(DAG_node const *to_node, DAG_node const *from_node) = 0;


    /// ================================================================================

    /// Field selectors, can be or'ed together to form a bit mask.
    enum Field_selector {
        FS_NONE                                   = 0x0000, ///< no fields copied
        //
        FS_MATERIAL_THIN_WALLED                   = 0x0001, ///< thin_walled field
        //
        FS_MATERIAL_SURFACE_SCATTERING            = 0x0002, ///< surface.scattering field
        FS_MATERIAL_SURFACE_EMISSION_EMISSION     = 0x0004, ///< surface.emission.emission field
        FS_MATERIAL_SURFACE_EMISSION_INTENSITY    = 0x0008, ///< surface.emission.intensity field
        FS_MATERIAL_SURFACE_EMISSION_MODE         = 0x0010, ///< surface.emission.mode field
        FS_MATERIAL_SURFACE_EMISSION              = 0x001c, ///< all surface.emission fields
        FS_MATERIAL_SURFACE                       = 0x001e, ///< all surface fields
        //
        FS_MATERIAL_BACKFACE_SCATTERING           = 0x0020, ///< backface.scattering field
        FS_MATERIAL_BACKFACE_EMISSION_EMISSION    = 0x0040, ///< backface.emission.emission field
        FS_MATERIAL_BACKFACE_EMISSION_INTENSITY   = 0x0080, ///< backface.emission.intensity field
        FS_MATERIAL_BACKFACE_EMISSION_MODE        = 0x0100, ///< backface.emission.mode field
        FS_MATERIAL_BACKFACE_EMISSION             = 0x01c0, ///< all backface.emission fields
        FS_MATERIAL_BACKFACE                      = 0x01e0, ///< all backface fields
        //
        FS_MATERIAL_IOR                           = 0x0200, ///< ior field
        //
        FS_MATERIAL_VOLUME_SCATTERING             = 0x0400, ///< volume.scattering field
        FS_MATERIAL_VOLUME_ABSORPTION_COEFFICIENT = 0x0800, ///< volume.absorption_coefficient field
        FS_MATERIAL_VOLUME_SCATTERING_COEFFICIENT = 0x1000, ///< volume.scattering_coefficient field
        FS_MATERIAL_VOLUME                        = 0x1c00, ///< all volume fields
        //
        FS_MATERIAL_GEOMETRY_DISPLACEMENT         = 0x2000, ///< geometry.displacement field
        FS_MATERIAL_GEOMETRY_CUTOUT_OPACITY       = 0x4000, ///< geometry.cutout_opacity field
        FS_MATERIAL_GEOMETRY_NORMAL               = 0x8000, ///< geometry.normal field
        FS_MATERIAL_GEOMETRY                      = 0xe000, ///< all geometry fields
        FS_MATERIAL_HAIR                          = 0x10000 ///< hair field
    };

    /// Returns a new material instance as a merge of two material instances based
    /// on a material field selection mask choosing the top-level material fields
    /// between the two materials.
    ///
    /// \param m0    the material instance whose fields are chosen if the mask bit is 0.
    /// \param m1    the material instance whose fields are chosen if the mask bit is 1.
    /// \param field_selector    mask to select the fields from m0 or m1 respectively.
    ///
    /// \return a new compiled material instance.
    virtual IMaterial_instance *merge_materials(
        IMaterial_instance const *m0,
        IMaterial_instance const *m1,
        Field_selector                                field_selector) = 0;

    /// Create a constant.
    ///
    /// \param  value       The value of the constant.
    ///
    /// \returns            The created constant.
    virtual DAG_constant const *create_constant(
        IValue const *value) = 0;

    /// Create a temporary reference.
    ///
    /// \param node         The DAG node that is "named" by this temporary.
    /// \param index        The index of the temporary.
    ///
    /// \returns            The created temporary reference.
    virtual DAG_temporary const *create_temporary(DAG_node const *node, int index) = 0;

    /// Create a call.
    ///
    /// \param  name            The name of the called function.
    /// \param  sema            The semantics of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the function.
    ///
    /// \returns                The created call or an equivalent expression.
    virtual DAG_node const *create_call(
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        int                           num_call_args,
        IType const                   *ret_type) = 0;

    /// Create a function call for a non-overloaded function. All parameter
    /// and return types are deduced from the function definition.
    ///
    /// \param  name            The name of the called function, e.g., "::state::normal".
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    ///
    /// \returns                The created call or an equivalent expression.
    virtual DAG_node const *create_function_call(
        char const             *name,
        DAG_node const * const call_args[],
        size_t                 num_call_args) = 0;

    /// Create a 1-, 2-, 3-, or 4-mixer call, with 2, 4, 6, or 8 parameters respectively.
    virtual DAG_node const *create_mixer_call(
        DAG_call::Call_argument const call_args[],
        int                           num_call_args) = 0;

    /// Create a 1-, 2-, 3-, or 4-color-mixer call, with 2, 4, 6, or 8 parameters respectively.
    virtual DAG_node const *create_color_mixer_call(
        DAG_call::Call_argument const call_args[],
        int                           num_call_args) = 0;

    /// Create a parameter reference.
    ///
    /// \param  type        The type of the parameter
    /// \param  index       The index of the parameter.
    ///
    /// \returns            The created parameter reference.
    virtual DAG_parameter const *create_parameter(IType const *type, int index) = 0;

    /// Get the type factory associated with this expression factory.
    ///
    /// \returns            The type factory.
    virtual IType_factory *get_type_factory() = 0;

    /// Get the value factory associated with this expression factory.
    ///
    /// \returns            The value factory.
    virtual IValue_factory *get_value_factory() = 0;

    /// Supported unary operators in the DAG representation.
    enum Unary_operator {
        OK_BITWISE_COMPLEMENT   = IExpression::OK_BITWISE_COMPLEMENT,
        OK_LOGICAL_NOT          = IExpression::OK_LOGICAL_NOT,
        OK_POSITIVE             = IExpression::OK_POSITIVE,
        OK_NEGATIVE             = IExpression::OK_NEGATIVE,
        OK_PRE_INCREMENT        = IExpression::OK_PRE_INCREMENT,
        OK_PRE_DECREMENT        = IExpression::OK_PRE_DECREMENT,
        OK_POST_INCREMENT       = IExpression::OK_POST_INCREMENT,
        OK_POST_DECREMENT       = IExpression::OK_POST_DECREMENT
    };

    /// Supported binary operators in the DAG representation.
    enum Binary_operator {
        OK_SELECT               = IExpression_binary::OK_SELECT,                ///< "."
        OK_ARRAY_INDEX          = IExpression_binary::OK_ARRAY_INDEX,           ///< "[]"
        OK_MULTIPLY             = IExpression_binary::OK_MULTIPLY,              ///< "*"
        OK_DIVIDE               = IExpression_binary::OK_DIVIDE,                ///< "/"
        OK_MODULO               = IExpression_binary::OK_MODULO,                ///< "%"
        OK_PLUS                 = IExpression_binary::OK_PLUS,                  ///< "+"
        OK_MINUS                = IExpression_binary::OK_MINUS,                 ///< "-"
        OK_SHIFT_LEFT           = IExpression_binary::OK_SHIFT_LEFT,            ///< "<<"
        OK_SHIFT_RIGHT          = IExpression_binary::OK_SHIFT_RIGHT,           ///< ">>"
        OK_UNSIGNED_SHIFT_RIGHT = IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT,  ///< ">>>"
        OK_LESS                 = IExpression_binary::OK_LESS,                  ///< "<"
        OK_LESS_OR_EQUAL        = IExpression_binary::OK_LESS_OR_EQUAL,         ///< "<="
        OK_GREATER_OR_EQUAL     = IExpression_binary::OK_GREATER_OR_EQUAL,      ///< ">="
        OK_GREATER              = IExpression_binary::OK_GREATER,               ///< ">"
        OK_EQUAL                = IExpression_binary::OK_EQUAL,                 ///< "=="
        OK_NOT_EQUAL            = IExpression_binary::OK_NOT_EQUAL,             ///< "!="
        OK_BITWISE_AND          = IExpression_binary::OK_BITWISE_AND,           ///< "&"
        OK_BITWISE_XOR          = IExpression_binary::OK_BITWISE_XOR,           ///< "^"
        OK_BITWISE_OR           = IExpression_binary::OK_BITWISE_OR,            ///< "|"
        OK_LOGICAL_AND          = IExpression_binary::OK_LOGICAL_AND,           ///< "&&"
        OK_LOGICAL_OR           = IExpression_binary::OK_LOGICAL_OR,            ///< "||"
    };

    /// Return the type for \::df::bsdf_component
    virtual IType const *get_bsdf_component_type() = 0;

    /// Return the type for \::df::edf_component
    virtual IType const *get_edf_component_type() = 0;

    /// Return the type for \::df::vdf_component
    virtual IType const *get_vdf_component_type() = 0;

    /// Return the type for \::df::bsdf_component
    virtual IType const *get_bsdf_component_array_type( int n_values) = 0;

    /// Return the type for \::df::edf_component
    virtual IType const *get_edf_component_array_type( int n_values) = 0;

    /// Return the type for \::df::vdf_component
    virtual IType const *get_vdf_component_array_type( int n_values) = 0;

    /// Return the type for \::df::color_bsdf_component
    virtual IType const *get_color_bsdf_component_type() = 0;

    /// Return the type for \::df::color_edf_component
    virtual IType const *get_color_edf_component_type() = 0;

    /// Return the type for \::df::color_vdf_component
    virtual IType const *get_color_vdf_component_type() = 0;

    /// Return the type for \::df::color_bsdf_component
    virtual IType const *get_color_bsdf_component_array_type( int n_values) = 0;

    /// Return the type for \::df::color_edf_component
    virtual IType const *get_color_edf_component_array_type( int n_values) = 0;

    /// Return the type for \::df::color_vdf_component
    virtual IType const *get_color_vdf_component_array_type( int n_values) = 0;

    /// Return the type for bool
    virtual IType const *get_bool_type() = 0;

    /// Creates an operator, handles types.
    ///
    /// \param op        the operator
    /// \param o         the operand
    ///
    /// \returns a DAG representing l op r
    virtual DAG_node const *create_unary(
        Unary_operator op,
        DAG_node const  *o) = 0;

    /// Creates an operator, handles types.
    ///
    /// \param op        the operator
    /// \param l         the left operand
    /// \param r         the right operand
    ///
    /// \returns a DAG representing l op r
    virtual DAG_node const *create_binary(
        Binary_operator op,
        DAG_node const  *l,
        DAG_node const  *r) = 0;

    /// Creates a ternary operator.
    ///
    /// \param cond      the condition
    /// \param t_expr    the true expression
    /// \param f_expr    the false expression
    ///
    /// \returns a DAG representing cond ? t_expr : f_expr
    virtual DAG_node const *create_ternary(
        DAG_node const *cond,
        DAG_node const *t_expr,
        DAG_node const *f_expr) = 0;

    /// Creates a SELECT operator on a struct or vector.
    ///
    /// \param s       a node producing a struct typed result
    /// \param member  the name of the member to select
    virtual DAG_node const *create_select(
        DAG_node const *s,
        char const     *member) = 0;

    /// Creates an array constructor.
    ///
    /// \param elem_type  the element type of the array, might be NULL iff n_values > 0
    /// \param values     the array elements
    /// \param n_values   number of values
    ///
    /// \note the element type cannot be derived from the values for zero-length arrays
    virtual DAG_node const *create_array(
        IType const            *elem_type,
        DAG_node const * const values[],
        size_t                 n_values) = 0;

    /// Creates a boolean constant.
    virtual DAG_constant const *create_bool_constant(bool f) = 0;

    /// Creates an integer constant.
    virtual DAG_constant const *create_int_constant(int i) = 0;

    /// Creates a constant of the predefined intensity_mode enum.
    ///
    /// \param i  the index of the enum value
    virtual DAG_constant const *create_emission_enum_constant(int i) = 0;

    /// Creates a constant of the df::scatter_mode enum.
    ///
    /// \param i  the index of the enum value
    virtual DAG_constant const *create_scatter_enum_constant(int i) = 0;

    /// Creates a constant of the tex::wrap_mode enum.
    ///
    /// \param i  the index of the enum value
    virtual DAG_constant const *create_wrap_mode_enum_constant(int i) = 0;

    /// Creates a floating point constant.
    virtual DAG_constant const *create_float_constant(float f) = 0;

    /// Creates a float3 constant.
    virtual DAG_constant const *create_float3_constant(float x, float y, float z) = 0;

    /// Creates a RGB color constant.
    virtual DAG_constant const *create_color_constant( float r, float g, float b) = 0;

    /// Creates a RGB color constant of the global material IOR value.
    virtual DAG_constant const *create_global_ior() = 0;

    /// Creates a float constant of the global material IOR green value.
    virtual DAG_constant const *create_global_float_ior() = 0;

    /// Creates a string constant.
    virtual DAG_constant const *create_string_constant(char const *s) = 0;

    /// Creates an invalid bsdf.
    virtual DAG_constant const *create_bsdf_constant() = 0;

    /// Creates an invalid edf.
    virtual DAG_constant const *create_edf_constant() = 0;

    /// Creates an invalid vdf.
    virtual DAG_constant const *create_vdf_constant() = 0;

    /// Creates an invalid hair_bsdf.
    virtual DAG_constant const *create_hair_bsdf_constant() = 0;

    /// Create a bsdf_component for a mixer; can be a call or a constant.
    virtual DAG_node const *create_bsdf_component(
        DAG_node const* weight_arg,
        DAG_node const* bsdf_arg) = 0;

    /// Create a edf_component for a mixer; can be a call or a constant.
    virtual DAG_node const *create_edf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg) = 0;

    /// Create a vdf_component for a mixer; can be a call or a constant.
    virtual DAG_node const *create_vdf_component(
        DAG_node const* weight_arg,
        DAG_node const* vdf_arg) = 0;

    /// Create a bsdf_color_component for a color mixer; can be a call or a constant.
    virtual DAG_node const *create_color_bsdf_component(
        DAG_node const* weight_arg,
        DAG_node const* bsdf_arg) = 0;

    /// Create a edf_color_component for a color mixer; can be a call or a constant.
    virtual DAG_node const *create_color_edf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg) = 0;

    /// Create a vdf_color_component for a color mixer; can be a call or a constant.
    virtual DAG_node const *create_color_vdf_component(
        DAG_node const* weight_arg,
        DAG_node const* edf_arg) = 0;

    /// Create a constant node for a given type and value.
    virtual DAG_constant const* mk_constant( const char* const_type, const char* value) = 0;

    /// Create DAG_node's for possible default values of Node_types parameter.
    virtual DAG_node const* mk_default( const char* param_type, const char* param_default) = 0;

    /// Returns the argument count if node is non-null and of the call kind or a compound constant,
    /// and 0 otherwise.
    virtual size_t get_compound_argument_size(DAG_node const* node) = 0;

    /// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
    /// and NULL otherwise.
    virtual DAG_node const *get_compound_argument(DAG_node const* node, size_t i) = 0;

    /// Return the i-th argument if node is non-null and of the call kind, or a compound constant,
    /// and NULL otherwise; remaps index for special case handling of mixers and parameter
    /// order of glossy BSDFs.
    virtual DAG_node const *get_remapped_argument(DAG_node const* node, size_t i) = 0;

    /// Returns the name of the i-th parameter of node, or NULL if there is none or node is NULL.
    virtual char const *get_compound_parameter_name(DAG_node const *node, size_t i) const = 0;

    /// Returns true if node evaluates to true
    virtual bool eval_if( DAG_node const* node) = 0;

    /// Returns true if node is not evaluating to false, i.e., it either evaluates
    /// to true or cannot be evaluated.
    virtual bool eval_maybe_if( DAG_node const* node) = 0;

    /// Compute the node selector for the matcher, either the semantic for a DAG_call
    /// node, or one of the Distiller_extended_node_semantics covering DAG_constant
    /// of type bsdf, edf or vdf respectively, or for DAG_constant's and DAG_call's of
    /// one of the material structs, and selectors for mix_1, mix_2, mix_3, mix_4,
    /// clamped_mix_1, ..., as well as a special selector for local_normal.
    /// All other nodes return 0.
    virtual int get_selector( DAG_node const* node) const = 0;

    /// Checks recursively for all call nodes if the property test_fct returns true.
    virtual bool all_nodes(
        IRule_matcher::Checker_function test_fct,
        DAG_node const *node) = 0;

    /// Set the normalization of mixer node flag and return its previous value.
    virtual bool set_normalize_mixers( bool new_value) = 0;

    /// Normalize mixer nodes and set respective flag to keep them normalized
    ///
    /// \param inst           a compiled material instance
    /// \param event_handler  if non-NULL, a event handler to report events during processing
    /// \param options        options for this rule set, currently none used.
    /// \param error          error codes reported back to the API
    ///
    /// \return a new compiled material
    virtual IMaterial_instance *normalize_mixers(
        IMaterial_instance const *inst,
        IRule_matcher_event                           *event_handler,
        const mi::mdl::Distiller_options              *options,
        mi::Sint32                                    &error) = 0;

    /// Immediately deletes this distiller plugin API
    virtual void release() const  = 0;

    virtual void debug_node(IOutput_stream *outs, DAG_node const *node) = 0;
};

} // mdl
} // mi

#endif // MDL_DISTILLER_PLUGIN_API_H
