/***************************************************************************************************
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief      MDL module builder

#ifndef MI_NEURAYLIB_IMDL_MODULE_BUILDER_H
#define MI_NEURAYLIB_IMDL_MODULE_BUILDER_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/version.h> // for MI_NEURAYLIB_DEPRECATED_ENUM_VALUE

namespace mi {

class IArray;

namespace neuraylib {

class IAnnotation_block;
class IAnnotation_list;
class IExpression;
class IExpression_list;
class IMdl_execution_context;

/** \addtogroup mi_neuray_mdl_misc
@{
*/

/// The module builder allows to create new MDL modules.
///
/// \see #mi::neuraylib::IMdl_factory::create_module_builder()
class IMdl_module_builder: public
    base::Interface_declare<0x2357f2f8,0x4428,0x47e5,0xaa,0x92,0x97,0x98,0x25,0x5d,0x26,0x57>
{
public:
    /// Adds a variant to the module.
    ///
    /// \param name                    The simple name of the material or function variant.
    /// \param prototype_name          The DB name of the prototype of the new variant.
    /// \param defaults                Additional/new default expressions to set. These defaults
    ///                                are used as arguments in the call expression that defines
    ///                                the variant. An empty expression list implicitly uses the
    ///                                existing defaults, whereas \c nullptr explicitly replicates
    ///                                the existing defaults of the original material or function.
    ///                                Feasible sub-expression kinds: constants and calls.
    /// \param annotations             Annotations to set. If \c nullptr, the annotations of the
    ///                                original material or function are used. Pass an empty block
    ///                                to remove all annotations.
    /// \param return_annotations      Return annotations to set. If \c nullptr, the annotations of
    ///                                the original material or function are used. Pass an empty
    ///                                block to remove all annotations. Materials require \c
    ///                                nullptr or an empty annotation block here.
    /// \param is_exported             Indicates whether the variant will have the 'export' keyword.
    /// \param is_declarative          Indicates whether the variant will have the 'declarative'
    ///                                keyword.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    /// \return                        0 in case of success, or -1 in case of failure.
    virtual Sint32 add_variant(
        const char* name,
        const char* prototype_name,
        const IExpression_list* defaults,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        bool is_declarative,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline Sint32 add_variant(
        const char* name,
        const char* prototype_name,
        const IExpression_list* defaults,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        IMdl_execution_context* context)
    {
        return add_variant(
            name,
            prototype_name,
            defaults,
            annotations,
            return_annotations,
            is_exported,
            false,
            context);
    }
#endif // MI_NEURAYLIB_DEPRECATED_15_0

    /// Adds a material or function to the module.
    ///
    /// \param name                    The simple name of the material or function.
    /// \param body                    The body of the new material or function (a constant, direct
    ///                                call, or parameter reference). Feasible sub-expression
    ///                                kinds: constants, direct calls, parameter references, and
    ///                                temporary references.
    /// \param temporaries             Temporaries referenced by the body. Feasible sub-expression
    ///                                kinds: constants, direct calls, parameter references, and
    ///                                temporary references with smaller index. The names
    ///                                associated with the expressions in the expressions list are
    ///                                not relevant (but need to unique as usual). Can be
    ///                                \c nullptr (treated like an empty expression list).
    /// \param parameters              Types and names of the parameters. Can be \c nullptr (treated
    ///                                like an empty parameter list).
    /// \param defaults                Default values. Can be \c nullptr or incomplete. Feasible
    ///                                sub-expression kinds: constants, calls, and direct calls.
    /// \param parameter_annotations   Parameter annotations. Can be \c nullptr or incomplete.
    /// \param annotations             Annotations of the material or function. Can be \c nullptr.
    /// \param return_annotations      Return annotations of the function. Can be \c nullptr for
    ///                                functions, must be \c nullptr for materials.
    /// \param is_exported             Indicates whether the material or function will have the
    ///                                'export' keyword.
    /// \param is_declarative          Indicates whether the material or function will have the
    ///                                'declarative' keyword.
    /// \param frequency_qualifier     The frequency qualifier for functions, or
    ///                                #mi::neuraylib::IType::MK_NONE for materials.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_function(
        const char* name,
        const IExpression* body,
        const IExpression_list* temporaries,
        const IType_list* parameters,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        bool is_declarative,
        IType::Modifier frequency_qualifier,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline Sint32 add_function(
        const char* name,
        const IExpression* body,
        const IType_list* parameters,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        const IAnnotation_block* return_annotations,
        bool is_exported,
        IType::Modifier frequency_qualifier,
        IMdl_execution_context* context)
    {
        return add_function(
            name, body, 0, parameters, defaults, parameter_annotations, annotations,
            return_annotations, is_exported, false, frequency_qualifier, context);
    }
#endif // MI_NEURAYLIB_DEPRECATED_15_0

    /// Adds an annotation to the module.
    ///
    /// \param name                    The simple name of the annotation.
    /// \param parameters              Types and names of the parameters. Can be \c nullptr (treated
    ///                                like an empty parameter list).
    /// \param defaults                Default values. Can be \c nullptr or incomplete. Feasible
    ///                                sub-expression kinds: constants, calls, and direct calls.
    /// \param parameter_annotations   Parameter annotations. Can be \c nullptr or incomplete.
    /// \param annotations             Annotations of the annotation. Can be \c nullptr.
    /// \param is_exported             Indicates whether the annotation will have the 'export'
    ///                                keyword.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_annotation(
        const char* name,
        const IType_list* parameters,
        const IExpression_list* defaults,
        const IAnnotation_list* parameter_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        IMdl_execution_context* context) = 0;

    /// Adds a struct category the module.
    ///
    /// \param name                    The simple name of the struct category.
    /// \param annotations             Annotations of the struct category. Can be \c nullptr.
    /// \param is_exported             Indicates whether the struct category will have the 'export'
    ///                                keyword.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_struct_category(
        const char* name,
        const IAnnotation_block* annotations,
        bool is_exported,
        IMdl_execution_context* context) = 0;

    /// Adds an enum type to the module.
    ///
    /// \note Changing a particular enum type, i.e., removing it and re-adding it differently, is
    ///       \em not supported.
    ///
    /// \param name                    The simple name of the enum type.
    /// \param enumerators             Enumerators of the enum type. Must not be empty. Feasible
    ///                                sub-expression kinds: constants and direct calls.
    /// \param enumerator_annotations  Enumerator annotations. Can be \c nullptr or incomplete.
    /// \param annotations             Annotations of the enum type. Can be \c nullptr.
    /// \param is_exported             Indicates whether the enum type will have the 'export'
    ///                                keyword.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_enum_type(
        const char* name,
        const IExpression_list* enumerators,
        const IAnnotation_list* enumerator_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        IMdl_execution_context* context) = 0;

    /// Adds a struct type to the module.
    ///
    /// \note Changing a particular struct type, i.e., removing it and re-adding it differently, is
    ///       \em not supported.
    ///
    /// \param name                    The simple name of the enum type.
    /// \param fields                  Fields of the struct type. Must not be empty.
    /// \param field_defaults          Defaults of the struct fields. Can be \c nullptr or
    ///                                incomplete. Feasible sub-expression kinds: constants and
    ///                                direct calls.
    /// \param field_annotations       Field annotations of the struct type. Can be \c nullptr or
    ///                                incomplete.
    /// \param annotations             Annotations of the struct type. Can be \c nullptr.
    /// \param is_exported             Indicates whether the struct type will have the 'export'
    ///                                keyword.
    /// \param is_declarative          Indicates whether the struct type will have the 'declarative'
    ///                                keyword.
    /// \param struct_category         The corresponding struct category. Can be \c nullptr.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_struct_type(
        const char* name,
        const IType_list* fields,
        const IExpression_list* field_defaults,
        const IAnnotation_list* field_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        bool is_declarative,
        const IStruct_category* struct_category,
        IMdl_execution_context* context) = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline Sint32 add_struct_type(
        const char* name,
        const IType_list* fields,
        const IExpression_list* field_defaults,
        const IAnnotation_list* field_annotations,
        const IAnnotation_block* annotations,
        bool is_exported,
        IMdl_execution_context* context)
    {
        return add_struct_type(
            name,
            fields,
            field_defaults,
            field_annotations,
            annotations,
            is_exported,
            false,
            0,
            context);
    }
#endif // MI_NEURAYLIB_DEPRECATED_15_0

    /// Adds a constant to the module.
    ///
    /// \param name                    The simple name of the constant.
    /// \param expr                    The value of the constant.
    ///                                Feasible sub-expression kinds: constants and direct calls.
    /// \param annotations             Annotations of the constant. Can be \c nullptr.
    /// \param is_exported             Indicates whether the constant will have the 'export'
    ///                                keyword.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 add_constant(
        const char* name,
        const IExpression* expr,
        const IAnnotation_block* annotations,
        bool is_exported,
        IMdl_execution_context* context) = 0;

    /// Sets the annotations of the module itself.
    ///
    /// \param annotations             Annotations of the module. Pass \c nullptr to remove existing
    ///                                annotations.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 set_module_annotations(
        const IAnnotation_block* annotations, IMdl_execution_context* context) = 0;

    /// Removes a material, function, enum or struct type from the module.
    ///
    /// \param name                    The simple name of material, function, enum or struct type to
    ///                                be removed.
    /// \param index                   The index of the function with the given name to be removed.
    ///                                Used to distinguish overloads of functions. Zero for
    ///                                materials, enum or struct types.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    virtual Sint32 remove_entity(
        const char* name, Size index, IMdl_execution_context* context) = 0;

    /// Clears the module, i.e., removes all entities from the module.
    virtual Sint32 clear_module( IMdl_execution_context* context) = 0;

    /// Analyzes which parameters need to be uniform.
    ///
    /// Note that the method can fail if the graph to be analyzed is invalid and/or never uniform
    /// independent of uniform modifiers of parameter types.
    ///
    /// <table>
    /// <tr>
    ///     <th>root_expr_uniform is \c false</th>
    ///     <th>root_expr_uniform is \c true</th>
    ///     <th>Interpretation</th>
    /// </tr>
    /// <tr><td>failure</td><td>failure</td><td>The graph is invalid.</td></tr>
    /// <tr><td>failure</td><td>success</td><td>This case is not possible.</td></tr>
    /// <tr><td>success</td><td>failure</td><td>The graph is never uniform.</td></tr>
    /// <tr><td>success</td><td>success</td><td>The graph is uniform if the parameters returned in
    ///                                         the \c true case are made uniform.</td></tr>
    /// </table>
    ///
    /// If the graph should be uniform, and you cannot rule out invalid graphs, then you might want
    /// to invoke this method first with \p root_expr_uniform set to \c false to check for
    /// validity. If that method succeeds, you can then call it again with \p root_expr_uniform set
    /// to \c true to obtain the constraints on the parameters.
    ///
    /// \param root_expr               Root expression of the graph, i.e., the body of the new
    ///                                material or function. Feasible sub-expression kinds:
    ///                                constants, direct calls, and parameter references.
    /// \param root_expr_uniform       Indicates whether the root expression should be uniform.
    /// \param context                 The execution context can be used to pass options and to
    ///                                retrieve error and/or warning messages. Can be \c nullptr.
    /// \return                        Returns an array of boolean values indicating which
    ///                                parameters need to be uniform (or \c nullptr in case of
    ///                                errors). The array indices match the indices of the
    ///                                parameter references. The array might be shorter than
    ///                                expected if trailing parameters are not referenced by \p
    ///                                root_expr.
    virtual const IArray* analyze_uniform(
        const IExpression* root_expr, bool root_expr_uniform, IMdl_execution_context* context) = 0;
};

/**@}*/ // end group mi_neuray_mdl_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_MODULE_BUILDER_H
