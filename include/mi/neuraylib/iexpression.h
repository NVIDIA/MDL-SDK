/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Expressions of the MDL type system

#ifndef MI_NEURAYLIB_IEXPRESSION_H
#define MI_NEURAYLIB_IEXPRESSION_H

#include <mi/neuraylib/ivalue.h>
#include <mi/neuraylib/version.h>

namespace mi {

class IString;

namespace neuraylib {

class IAnnotation;
class IAnnotation_block;
class IExpression_list;

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// The MDL version.
enum Mdl_version {
    MDL_VERSION_1_0,                       ///< MDL version 1.0
    MDL_VERSION_1_1,                       ///< MDL version 1.1
    MDL_VERSION_1_2,                       ///< MDL version 1.2
    MDL_VERSION_1_3,                       ///< MDL version 1.3
    MDL_VERSION_1_4,                       ///< MDL version 1.4
    MDL_VERSION_1_5,                       ///< MDL version 1.5
    MDL_VERSION_1_6,                       ///< MDL version 1.6
    MDL_VERSION_1_7,                       ///< MDL version 1.7
    MDL_VERSION_1_8,                       ///< MDL version 1.8
    MDL_VERSION_1_9,                       ///< MDL version 1.9
    MDL_VERSION_EXP,                       ///< MDL experimental features.
    MDL_VERSION_LATEST = MDL_VERSION_1_9,  ///< Latest MDL version
    MDL_VERSION_INVALID = 0xffffffffU,     ///< Invalid MDL version
    MDL_VERSION_FORCE_32_BIT = 0xffffffffU // Undocumented, for alignment only
};

/// The interface to MDL expressions.
///
/// Expressions can be created using the expression factory #mi::neuraylib::IExpression_factory.
class IExpression : public
    mi::base::Interface_declare<0x0f4a7542,0x9b27,0x4924,0xbd,0x8d,0x82,0xe3,0xa9,0xa7,0xa9,0xd6>
{
public:
    /// The possible kinds of expressions.
    enum Kind {
        /// A constant expression. See #mi::neuraylib::IExpression_constant.
        EK_CONSTANT,
        /// An indirect call expression. See #mi::neuraylib::IExpression_call.
        EK_CALL,
        /// A parameter reference expression. See #mi::neuraylib::IExpression_parameter.
        EK_PARAMETER,
        /// A direct call expression. See #mi::neuraylib::IExpression_direct_call.
        EK_DIRECT_CALL,
        /// A temporary reference expression. See #mi::neuraylib::IExpression_temporary.
        EK_TEMPORARY,
        //  Undocumented, for alignment only.
        EK_FORCE_32_BIT = 0xffffffffU
    };

    /// Returns the kind of this expression.
    virtual Kind get_kind() const = 0;

    /// Returns the type of this expression.
    virtual const IType* get_type() const = 0;

    /// Returns the type of this expression.
    template <class T>
    const T* get_type() const
    {
        const IType* ptr_type = get_type();
        if( !ptr_type)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
        ptr_type->release();
        return ptr_T;
    }
};

mi_static_assert( sizeof( IExpression::Kind) == sizeof( Uint32));

/// A constant expression.
///
/// Constant expressions represent values (see #mi::neuraylib::IValue). They appear basically
/// everywhere where expressions are used.
class IExpression_constant : public
    mi::base::Interface_declare<0x9da8d465,0x4058,0x46cb,0x83,0x6e,0x0e,0x38,0xa6,0x7f,0xcd,0xef,
                                neuraylib::IExpression>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = EK_CONSTANT;

    /// Returns the value of the constant.
    virtual const IValue* get_value() const = 0;

    /// Returns the value of the constant.
    template <class T>
    const T* get_value() const
    {
        const IValue* ptr_value = get_value();
        if( !ptr_value)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Returns the value of the constant.
    virtual IValue* get_value() = 0;

    /// Returns the value of the constant.
    template <class T>
    T* get_value()
    {
        IValue* ptr_value = get_value();
        if( !ptr_value)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_value->get_interface( typename T::IID()));
        ptr_value->release();
        return ptr_T;
    }

    /// Sets the value of the constant.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: The type of \p value does not match the type of the constant.
    virtual Sint32 set_value( IValue* value) = 0;
};

/// An indirect call expression.
///
/// This call expression is called \em indirect since it just references another DB element
/// representing the actual call. This contrasts with #mi::neuraylib::IExpression_direct_call for
/// \em direct call expressions. See below for a more detailed comparison.
///
/// Indirect call expressions appear as defaults of material or function definitions, as arguments
/// of material instances or function calls, and as defaults in the module builder.
///
/// \par Indirect vs direct calls
///
/// There are two different types of call expressions, \em indirect calls (represented by this
/// interface #mi::neuraylib::IExpression_call) and \em direct calls (represented by
/// #mi::neuraylib::IExpression_direct_call). Both have their disjoint use cases, and they never
/// appear together in the very same expression graph.
///
/// Indirect calls are used where the expression graph is frequently changed by the user, in
/// particular for arguments of function calls and material instances. These indirect calls simply
/// reference another DB element by name, which represents the actual call. Note that such an
/// argument might not just be a constant or a single call (with constants as arguments), but can be
/// a rather complex graph (so-called open material graphs). The indirection via DB elements allows
/// to access any such graph node directly (assuming its name is known) in order to modify it,
/// without traversing the entire expression graph first to reach a given node.
///
/// Direct calls are used where the expression graph is fixed (bodies of material and function
/// definitions, and compiled materials), or where such a graph is constructed once and not
/// modified later (bodies in the module builder). In these cases the indirection via DB elements
/// is not necessary and would be rather cumbersome.
///
/// Note that defaults use indirect calls, even though they are fixed. This exception was made to
/// stress the similarity to arguments, e.g., it allows to pass a default as a new argument to
/// #mi::neuraylib::IFunction_call::set_argument().
class IExpression_call : public
    mi::base::Interface_declare<0xcf625aec,0x8eb8,0x4743,0x9f,0xf6,0x76,0x82,0x2c,0x02,0x54,0xa3,
                                neuraylib::IExpression>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = EK_CALL;

    /// Returns the DB name of the referenced function call or material instance.
    virtual const char* get_call() const = 0;

    /// Sets the name of the referenced function call or material instance.
    ///
    /// \param name    The DB name of the function call or material instance.
    /// \return
    ///                -  0: Success.
    ///                - -1: Invalid parameter (\c NULL pointer).
    ///                - -2: There is no DB element with that name.
    ///                - -3: The DB element has not the correct type.
    ///                - -4: The return type of the DB element does not match the type of this
    ///                      expression.
    ///                - -5: The material instance or function call referenced by "name" is
    ///                      a parameter default and therefore cannot be used in a call.
    ///                - -6: The material instance or function call referenced by "name" is
    ///                      invalid and therefore cannot be used in a call.
    virtual Sint32 set_call( const char* name) = 0;
};

/// A parameter reference expression.
///
/// Parameter reference expressions are used for defaults that reference earlier parameters. For
/// example consider
/// \code
/// float foo(float a, float b, float c = b) { ... }
/// \endcode
/// The default for the parameter \c c is of type #mi::neuraylib::IExpression_parameter. Its index
/// is 1, equal to the parameter index of parameter \c b.
///
/// Parameter reference expressions appear as defaults of material or function definitions. They are
/// expanded during #mi::neuraylib::IFunction_definition::create_function_call(),
/// #mi::neuraylib::IFunction_call::reset_argument(), and
/// #mi::neuraylib::Definition_wrapper::create_instance().
class IExpression_parameter : public
    mi::base::Interface_declare<0x206c4319,0x0b53,0x45a7,0x86,0x07,0x29,0x98,0xb3,0x44,0x7f,0xaa,
                               neuraylib::IExpression>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = EK_PARAMETER;

    /// Returns the index of the referenced parameter.
    virtual Size get_index() const = 0;

    /// Sets the index of the referenced parameter.
    virtual void set_index( Size index) = 0;
};

/// A direct call expression.
///
/// This call expression is called \em direct since it directly represents the actual call (and not
/// simply references another DB element representing the actual call as for indirect call
/// expressions). See #mi::neuraylib::IExpression_call for of comparison of direct and indirect call
/// expressions.
///
/// Direct call expressions appear in fields and temporaries of compiled materials, in the bodies
/// of function and material definitions, and in bodies in the module builder.
class IExpression_direct_call : public
    mi::base::Interface_declare<0x9253c9d6,0xe162,0x4234,0xab,0x91,0x54,0xc1,0xe4,0x87,0x39,0x66,
                                neuraylib::IExpression>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = EK_DIRECT_CALL;

    /// Returns the DB name of the corresponding function or material definition.
    virtual const char* get_definition() const = 0;

    /// Returns the arguments of the direct call.
    virtual const IExpression_list* get_arguments() const = 0;
};

/// A temporary reference expression.
///
/// Temporary reference expressions are used to make the structure of DAGs (directed acyclic
/// graphs) more explicit. Consider a diamond-shaped graph with four nodes A, B, C, and D. Node A
/// references nodes B and C, and both nodes B and C reference node D.
///
/// \image html expressions1.png "Diamond-shaped graph"
/// \image latex expressions1.png "Diamond-shaped graph"
///
/// When traversing this graph starting at node A, it could be important to realize when the node D
/// is visited a second time (to avoid double work for node D, or to avoid misinterpreting the
/// graph as a tree).
///
/// Note that in this example we use letters to identify the nodes. But expressions themselves do
/// not have names. And the \neurayApiName does not guarantee identical pointer addresses for
/// repeated accesses to the (logically) same object. For example, repeated calls of
/// #mi::neuraylib::IExpression_direct_call::get_arguments() followed by
/// #mi::neuraylib::IExpression_list::get_expression() result in different pointers, although
/// logically the result is always the same (for the same input). Therefore, a different mechanism
/// is needed to recognize nodes that can be visited several times in a traversal.
///
/// This is where temporary references come into play. In an expression graph for the example above,
/// the nodes B and C do not point directly to node D, but to two temporary reference expressions,
/// both with the same index. The node D can then be retrieved by passing that index to
/// #mi::neuraylib::IFunction_definition::get_temporary() or
/// #mi::neuraylib::ICompiled_material::get_temporary().
///
/// \image html expressions2.png "Same graph, but with temporary references"
/// \image latex expressions2.png "Same graph, but with temporary references"
///
/// A common idiom is to traverse temporaries first, storing the result for each temporary, and to
/// use them when encountering the corresponding temporary reference. Alternatively, one can
/// traverse temporaries when encountered on-demand, store the result for each temporary, and reuse
/// the result of the first encounter of a particular temporary on later encounters of that
/// temporary.
///
/// Note that, despite the name #mi::neuraylib::IExpression_temporary, expressions of this type are
/// not the temporaries themselves, but \em references to temporaries. Temporaries can be nested,
/// i.e., a temporary with index i might contain temporary references for other temporaries with
/// indices up to i-1.
///
/// Temporary reference expressions appear in fields (and temporaries) of compiled materials, and in
/// the bodies (and temporaries) of function and material definitions.
class IExpression_temporary : public
    mi::base::Interface_declare<0xd91f484b,0xdbf8,0x4585,0x9d,0xab,0xba,0xd9,0x91,0x7f,0xe1,0x4c,
                                neuraylib::IExpression>
{
public:
    /// The kind of this subclass.
    static const Kind s_kind = EK_TEMPORARY;

    /// Returns the index of the referenced temporary.
    virtual Size get_index() const = 0;

    /// Sets the index of the referenced temporary.
    virtual void set_index( Size index) = 0;
};

/// An ordered collection of expressions identified by name or index.
///
/// Expression lists can be created with
/// #mi::neuraylib::IExpression_factory::create_expression_list().
class IExpression_list : public
    mi::base::Interface_declare<0x98ce8e89,0x9f23,0x45ec,0xa7,0xce,0x85,0x78,0x48,0x14,0x85,0x23>
{
public:
    /// Returns the number of elements.
    virtual Size get_size() const = 0;

    /// Returns the index for the given name, or -1 if there is no such expression.
    virtual Size get_index( const char* name) const = 0;

    /// Returns the name for the given index, or \c NULL if there is no such expression.
    virtual const char* get_name( Size index) const = 0;

    /// Returns the expression for \p index, or \c NULL if there is no such expression.
    ///
    /// This index-based overload is faster than the name-based overload
    /// #get_expression(const char*)const and should be preferred if the index is known.
    ///
    /// \note Expression lists for defaults and enable-if conditions might not contain an
    ///       expression for every parameter. Therefore, indices in such lists are not
    ///       guaranteed to match the parameter indices.
    virtual const IExpression* get_expression( Size index) const = 0;

    /// Returns the expression for \p index, or \c NULL if there is no such expression.
    ///
    /// This index-based overload is faster than the name-based overload
    /// #get_expression<T>(const char*)const and should be preferred if the index is known.
    ///
    /// \note Expression lists for defaults and enable-if conditions might not contain an
    ///       expression for every parameter. Therefore, indices in such lists are not
    ///       guaranteed to match the parameter indices.
    template <class T>
    const T* get_expression( Size index) const
    {
        const IExpression* ptr_expression = get_expression( index);
        if( !ptr_expression)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_expression->get_interface( typename T::IID()));
        ptr_expression->release();
        return ptr_T;
    }

    /// Returns the expression for \p name, or \c NULL if there is no such expression.
    ///
    /// The index-based overload #get_expression(const char*)const is faster than this name-based
    /// overload and should be preferred if the index is known.
    virtual const IExpression* get_expression( const char* name) const = 0;

    /// Returns the expression for \p name, or \c NULL if there is no such expression.
    ///
    /// The index-based overload #get_expression<T>(const char*)const is faster than this name-based
    /// overload and should be preferred if the index is known.
    template <class T>
    const T* get_expression( const char* name) const
    {
        const IExpression* ptr_expression = get_expression( name);
        if( !ptr_expression)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_expression->get_interface( typename T::IID()));
        ptr_expression->release();
        return ptr_T;
    }

    /// Sets an expression at a given index.
    ///
    /// This index-based overload is faster than the name-based overload
    /// #set_expression(const char*,const IExpression*) and should be preferred if the index is
    /// known.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_expression( Size index, const IExpression* expression) = 0;

    /// Sets an expression identified by name.
    ///
    /// The index-based overload #set_expression(const char*,const IExpression*) is faster than
    /// this name-based overload and should be preferred if the index is known.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is no expression mapped to \p name in the list.
    virtual Sint32 set_expression( const char* name, const IExpression* expression) = 0;

    /// Adds an expression at the end of the list.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is already an expression mapped to \p name in the list.
    virtual Sint32 add_expression( const char* name, const IExpression* expression) = 0;
};

/// An annotation definition.
///
class IAnnotation_definition : public
    mi::base::Interface_declare<0xa453318b,0xe056,0x4521,0x9f,0x3c,0x9d,0x5c,0x3,0x23,0x5f,0xb7>
{
public:

    /// All known semantics of annotation definitions.
    ///
    /// \note Do not rely on the numeric values of the enumerators since they may change without
    ///       further notice.
    enum Semantics
    {
        AS_UNKNOWN = 0,                          ///< Unknown semantics.

        /// This is the internal intrinsic() annotation.
        AS_INTRINSIC_ANNOTATION = 0x0100,
        AS_ANNOTATION_FIRST = AS_INTRINSIC_ANNOTATION,
        AS_THROWS_ANNOTATION,                    ///< This is the internal throws() annotation.
        AS_SINCE_ANNOTATION,                     ///< This is the internal since() annotation.
        AS_REMOVED_ANNOTATION,                   ///< This is the internal removed() annotation.
        AS_CONST_EXPR_ANNOTATION,                ///< This is the internal const_expr() annotation.
        AS_DERIVABLE_ANNOTATION,                 ///< This is the internal derivable() annotation.
        AS_NATIVE_ANNOTATION,                    ///< This is the internal native() annotation.

        AS_UNUSED_ANNOTATION,                    ///< This is the unused() annotation.
        AS_NOINLINE_ANNOTATION,                  ///< This is the noinline() annotation.
        AS_SOFT_RANGE_ANNOTATION,                ///< This is the soft_range() annotation.
        AS_HARD_RANGE_ANNOTATION,                ///< This is the hard_range() annotation.
        AS_HIDDEN_ANNOTATION,                    ///< This is the hidden() annotation.
        AS_DEPRECATED_ANNOTATION,                ///< This is the deprecated() annotation.
        AS_VERSION_NUMBER_ANNOTATION,            ///< This is the (old) version_number() annotation.
        AS_VERSION_ANNOTATION,                   ///< This is the version() annotation.
        AS_DEPENDENCY_ANNOTATION,                ///< This is the dependency() annotation.
        AS_UI_ORDER_ANNOTATION,                  ///< This is the ui_order() annotation.
        AS_USAGE_ANNOTATION,                     ///< This is the usage() annotation.
        AS_ENABLE_IF_ANNOTATION,                 ///< This is the enable_if() annotation.
        AS_THUMBNAIL_ANNOTATION,                 ///< This is the thumbnail() annotation.
        AS_DISPLAY_NAME_ANNOTATION,              ///< This is the display_name() annotation.
        AS_IN_GROUP_ANNOTATION,                  ///< This is the in_group() annotation.
        AS_DESCRIPTION_ANNOTATION,               ///< This is the description() annotation.
        AS_AUTHOR_ANNOTATION,                    ///< This is the author() annotation.
        AS_CONTRIBUTOR_ANNOTATION,               ///< This is the contributor() annotation.
        AS_COPYRIGHT_NOTICE_ANNOTATION,          ///< This is the copyright_notice() annotation.
        AS_CREATED_ANNOTATION,                   ///< This is the created() annotation.
        AS_MODIFIED_ANNOTATION,                  ///< This is the modified() annotation.
        AS_KEYWORDS_ANNOTATION,                  ///< This is the key_words() annotation.
        AS_ORIGIN_ANNOTATION,                    ///< This is the origin() annotation.
        AS_NODE_OUTPUT_PORT_DEFAULT_ANNOTATION,  ///< This is the node_output_port_default()
                                                 ///  annotation.

        AS_ANNOTATION_LAST = AS_NODE_OUTPUT_PORT_DEFAULT_ANNOTATION,
        AS_FORCE_32_BIT = 0xffffffffU            //   Undocumented, for alignment only.
    };

    /// Returns the DB name of the module containing this annotation definition.
    ///
    /// The type of the module is #mi::neuraylib::IModule.
    virtual const char* get_module() const = 0;

    /// Returns the MDL name of the annotation definition.
    virtual const char* get_name() const = 0;

    /// Returns the MDL name of the module containing this annotation definition.
    virtual const char* get_mdl_module_name() const = 0;

    /// Returns the simple MDL name of the annotation definition.
    ///
    /// The simple name is the last component of the MDL name, i.e., without any packages and
    /// scope qualifiers, and without the parameter type names.
    ///
    /// \return         The simple MDL name of the annotation definition.
    virtual const char* get_mdl_simple_name() const = 0;

    /// Returns the type name of the parameter at \p index.
    ///
    /// \note The type names provided here are substrings of the MDL name returned by #get_name().
    ///       They are provided here such that parsing of the MDL name is not necessary. However,
    ///       for most use cases it is strongly recommended to use #get_parameter_types() instead.
    ///
    /// \param index    The index of the parameter.
    /// \return         The type name of the parameter, or \c NULL if \p index is out of range.
    virtual const char* get_mdl_parameter_type_name( Size index) const = 0;

    /// Returns the semantic of this annotation definition.
    virtual Semantics get_semantic() const = 0;

    /// Indicates whether the annotation definition is exported by its module.
    virtual bool is_exported() const = 0;

    /// Returns the MDL version when this annotation definition was added and removed.
    ///
    /// \param[out] since     The MDL version in which this annotation definition was added. If the
    ///                       annotation definition does not belong to the standard library, the
    ///                       MDL version of the corresponding module is returned.
    /// \param[out] removed   The MDL version in which this annotation definition was removed, or
    ///                       mi::neuraylib::MDL_VERSION_INVALID if the annotation has not been
    ///                       removed so far or does not belong to the standard library.
    virtual void get_mdl_version( Mdl_version& since, Mdl_version& removed) const = 0;

    /// Returns the parameter count of the annotation definition.
    virtual Size get_parameter_count() const = 0;

    /// Returns the parameter name of the given index.
    ///
    /// \param index    The parameter index.
    /// \return         The name of the parameter or \c NULL if index
    ///                 is out of range.
    virtual const char* get_parameter_name( Size index) const = 0;

    /// Returns the parameter index of the given name.
    ///
    /// \param name     The parameter name.
    /// \return         The index of the parameter or \c -1 if there is no
    ///                 parameter of that \p name.
    virtual Size get_parameter_index( const char* name) const = 0;

    /// Returns the parameter types of the annotation definition.
    virtual const IType_list* get_parameter_types() const = 0;

    /// Returns the parameter defaults of the annotation definition.
    virtual const IExpression_list* get_defaults() const = 0;

    /// Returns the annotations of this definition or \c NULL if no
    /// annotations exist.
    virtual const IAnnotation_block* get_annotations() const = 0;

    /// Creates an annotation.
    ///
    /// \param arguments    The arguments for new annotation.
    /// \return             The created annotation or \c NULL if one of the arguments
    ///                     does not correspond to an actual parameter of the annotation or
    ///                     is not a constant expression.
    virtual const IAnnotation* create_annotation( const IExpression_list* arguments) const = 0;
};

mi_static_assert( sizeof( IAnnotation_definition::Semantics) == sizeof( Uint32));

/// An annotation is similar to a direct call expression, but without return type. Its definition
/// can be obtained by calling #mi::neuraylib::IAnnotation::get_definition().
///
/// Annotations can be created with #mi::neuraylib::IExpression_factory::create_annotation() or
/// #mi::neuraylib::IAnnotation_definition::create_annotation().
class IAnnotation : public
    mi::base::Interface_declare<0xa9c652e7,0x952e,0x4887,0x93,0xb4,0x55,0xc8,0x66,0xd0,0x1a,0x1f>
{
public:
    /// Returns the name of the annotation.
    virtual const char* get_name() const = 0;

    /// Sets the name of the annotation.
    virtual void set_name( const char* name) = 0;

    /// Returns the arguments of the annotation.
    ///
    /// The arguments of annotations are always constant expressions.
    virtual const IExpression_list* get_arguments() const = 0;

    /// Returns the definition of this annotation.
    virtual const IAnnotation_definition* get_definition() const = 0;
};

/// An annotation block is an array of annotations.
///
/// Annotation blocks can be created with
/// #mi::neuraylib::IExpression_factory::create_annotation_block().
class IAnnotation_block : public
    mi::base::Interface_declare<0x57b0ae97,0x0815,0x41e8,0x89,0xe7,0x16,0xa1,0x23,0x86,0x80,0x6e>
{
public:
    /// Returns the number of annotations in this block.
    virtual Size get_size() const = 0;

    /// Returns the annotation for \p index, or \c NULL if index is out of bounds.
    virtual const IAnnotation* get_annotation( Size index) const = 0;

    /// Sets an annotation block at a given index.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_annotation( Size index, const IAnnotation* annotation) = 0;

    /// Adds an annotation at the end of the annotation block.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    virtual Sint32 add_annotation( IAnnotation* annotation) = 0;
};

/// An ordered collection of annotation blocks identified by name or index.
///
/// Annotation lists can be created with
/// #mi::neuraylib::IExpression_factory::create_annotation_list().
class IAnnotation_list : public
    mi::base::Interface_declare<0x6c4663c2,0x112f,0x4eeb,0x81,0x60,0x41,0xa5,0xa6,0xfb,0x74,0x3c>
{
public:
    /// Returns the number of elements.
    virtual Size get_size() const = 0;

    /// Returns the index for the given name, or -1 if there is no such block.
    virtual Size get_index( const char* name) const = 0;

    /// Returns the name for the given index, or \c NULL if there is no such block.
    virtual const char* get_name( Size index) const = 0;

    /// Returns the annotation block for \p index, or \c NULL if there is no such block.
    virtual const IAnnotation_block* get_annotation_block( Size index) const = 0;

    /// Returns the annotation block for \p name, or \c NULL if there is no such block.
    virtual const IAnnotation_block* get_annotation_block( const char* name) const = 0;

    /// Sets an annotation block at a given index.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_annotation_block( Size index, const IAnnotation_block* block) = 0;

    /// Sets an annotation block identified by name.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is no annotation block mapped to \p name in the list.
    virtual Sint32 set_annotation_block( const char* name, const IAnnotation_block* block) = 0;

    /// Adds an annotation block at the end of the list.
    ///
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: There is already an annotation block mapped to \p name in the list.
    virtual Sint32 add_annotation_block( const char* name, const IAnnotation_block* block) = 0;
};

/// The interface for creating expressions.
///
/// An expression factory can be obtained from
/// #mi::neuraylib::IMdl_factory::create_expression_factory().
///
/// \see #mi::neuraylib::IType_factory, #mi::neuraylib::IValue_factory
class IExpression_factory : public
    mi::base::Interface_declare<0x9fd3b2d4,0xb5b8,0x4ccd,0x9b,0x5f,0x7b,0xd9,0x9d,0xeb,0x62,0x64>
{
public:
    /// \name Access to related factories.
    //@{

    /// Returns the value factory associated with this expression factory.
    virtual IValue_factory* get_value_factory() const = 0;

    //@}
    /// \name Creation of expressions and expression lists
    //@{

    /// Creates a constant (mutable).
    ///
    /// \param value        The value of the constant.
    /// \return             The created constant, or \c NULL in case of errors.
    virtual IExpression_constant* create_constant( IValue* value) const = 0;

    /// Creates a constant (const).
    ///
    /// \param value        The value of the constant.
    /// \return             The created constant.
    virtual const IExpression_constant* create_constant( const IValue* value) const = 0;

    /// Creates a call.
    ///
    /// \param name         The DB name of the referenced function call or material instance.
    /// \return             The created call, or \c NULL in case of errors.
    virtual IExpression_call* create_call( const char* name) const = 0;

    /// Creates a parameter reference.
    ///
    /// \param type         The type of the parameter.
    /// \param index        The index of the parameter.
    /// \return             The created parameter reference, or \c NULL in case of errors.
    virtual IExpression_parameter* create_parameter( const IType* type, Size index) const = 0;

    /// Creates a direct call.
    ///
    /// \param name         The DB name of the referenced function or material definition.
    /// \param arguments    The arguments of the created direct call. \n
    ///                     Arguments for parameters without default are mandatory, otherwise
    ///                     optional. The type of an argument must match the corresponding parameter
    ///                     type. Any argument missing in \p arguments will be set to the default of
    ///                     the corresponding parameter. \n
    ///                     Note that the expressions in \p arguments are copied. Valid
    ///                     subexpressions are constants, direct calls, and parameter references.
    ///                     operation is a deep copy, e.g., DB elements referenced in call
    ///                     expressions are also copied. \n
    ///                     \c NULL is a valid argument which is handled like an empty expression
    ///                     list.
    /// \param[out] errors  An optional pointer to an #mi::Sint32 to which an error code will be
    ///                     written. The error codes have the following meaning:
    ///                     -  0: Success.
    ///                     - -1: An argument for a non-existing parameter was provided in
    ///                           \p arguments.
    ///                     - -2: The type of an argument in \p arguments does not have the correct
    ///                           type.
    ///                     - -3: A parameter that has no default was not provided with an argument
    ///                           value.
    ///                     - -5: A parameter type is uniform, but the corresponding argument has a
    ///                           varying return type.
    ///                     - -6: An argument expression is not a constant, a direct call, nor a
    ///                           parameter.
    ///                     - -7: Invalid parameters (\c NULL pointer) or \p name is not a valid
    ///                           DB name of a function or material definition.
    ///                     - -8: One of the parameter types is uniform, but the corresponding
    ///                           argument or default is a call expression and the return type of
    ///                           the called function or material definition is effectively varying
    ///                           since the function or material definition itself is varying.
    ///                     - -9: The function or material definition is invalid due to a module
    ///                           reload.
    /// \return             The created call, or \c NULL in case of errors.
    virtual IExpression_direct_call* create_direct_call(
        const char* name, IExpression_list* arguments, Sint32* errors = 0) const = 0;

    /// Creates a temporary reference.
    ///
    /// \param type         The type of the temporary.
    /// \param index        The index of the temporary.
    /// \return             The created temporary reference, or \c NULL in case of errors.
    virtual IExpression_temporary* create_temporary( const IType* type, Size index) const = 0;

    /// Creates a new expression list.
    virtual IExpression_list* create_expression_list() const = 0;

    //@}
    /// \name Creation of annotations, annotation blocks, and annotations lists
    //@{

    /// Creates a new annotation.
    ///
    /// Returns \c NULL if one of the arguments is not a constant expression.
    virtual IAnnotation* create_annotation(
        const char* name, const IExpression_list* arguments) const = 0;

    /// Creates a new annotation block.
    virtual IAnnotation_block* create_annotation_block() const = 0;

    /// Creates a new annotation list.
    virtual IAnnotation_list* create_annotation_list() const = 0;

    //@}
    /// \name Cloning of expressions and expression lists
    //@{

    /// Clones the given expression.
    ///
    /// Note that referenced DB elements, e.g., resources in constant expressions, or function calls
    /// and material instances in call expressions, are not copied, but shared. Function calls and
    /// material instances that serve as default arguments, are copied, though.
    virtual IExpression* clone( const IExpression* expr) const = 0;

    /// Clones the given expression.
    ///
    /// Note that referenced DB elements, e.g., resources in constant expressions, or function calls
    /// and material instances in call expressions, are not copied, but shared. Function calls and
    /// material instances that serve as default arguments, are copied, though.
    template <class T>
    T* clone( const T* expr) const
    {
        IExpression* ptr_expr = clone( static_cast<const IExpression*>( expr));
        if( !ptr_expr)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_expr->get_interface( typename T::IID()));
        ptr_expr->release();
        return ptr_T;
    }

    /// Clones the given expression list.
    ///
    /// Note that referenced DB elements, e.g., resources in constant expressions, or function calls
    /// and material instances in call expressions, are not copied, but shared.
    virtual IExpression_list* clone( const IExpression_list* expression_list) const = 0;

    //@}
    /// \name Comparison of expression and expression lists
    //@{

    /// Various options for the comparison of expressions or expression lists.
    ///
    /// \see The \p flags parameter of #compare()
    enum Comparison_options {
        /// Default comparison options.
        DEFAULT_OPTIONS      = 0,
        /// This option indicates that call expressions should be compared for equality, not for
        /// identity. That is, the comparison is not done via
        /// #mi::neuraylib::IExpression::get_value(), but by traversing into the referenced
        /// function call, i.e., comparing the function definition reference and the arguments.
        /// This option is useful if you want to decide whether an argument is \em semantically
        /// equal to the corresponding default parameter.
        DEEP_CALL_COMPARISONS           = 1,
        /// This option indicates that all type aliases should be skipped before types of
        /// expression are compared. Defaults and argument might sometimes differ in explicit type
        /// modifiers, therefore this option is useful if you want to decide whether an argument is
        /// \em semantically equal to the corresponding default parameter.
        SKIP_TYPE_ALIASES               = 2,
        // Undocumented, for alignment only
        COMPARISON_OPTIONS_FORCE_32_BIT = 0xffffffffU
    };

    /// Compares two instances of #mi::neuraylib::IExpression.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IExpression is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Otherwise, the types of \p lhs and \p rhs are compared. If they are different, the result
    ///   is determined by that comparison.
    /// - Next, the kind of the expressions are compared. If they are different, the result is
    ///   determined by \c operator< on the #mi::neuraylib::IExpression::Kind values. Note that
    ///   setting #SKIP_TYPE_ALIASES in \p flags modifies this behavior.
    /// - Finally, the expressions are compared as follows:
    ///   - For constants the results is defined by comparing their values.
    ///   - For calls, the result is defined by comparison of the call reference (unless
    ///     #DEEP_CALL_COMPARISONS is set in \p flags). Note that the representation of this call
    ///     reference is an internal implementation detail, and the comparison result might have
    ///     the opposite sign as \c strcmp() on the strings returned by
    ///     #mi::neuraylib::IExpression_call::get_call().
    ///   - For parameter and temporary references, the results is defined by \c operator<() on the
    ///     indices.
    ///   - For indirect calls, first the definition reference is compared. Note that the
    ///     representation of this definition reference is an internal implementation detail, and
    ///     the comparison result might have the opposite sign as \c strcmp() on the strings
    ///     returned by #mi::neuraylib::IExpression_direct_call::get_definition(). If both indirect
    ///     call reference the same definition, then the result is defined by comparison of the
    ///     arguments.
    ///
    /// \param lhs          The left-hand side operand for the comparison.
    /// \param rhs          The right-hand side operand for the comparison.
    /// \param flags        A bitmask of flags of type #Comparison_options.
    /// \param epsilon      Maximum difference for floating point values to consider them as equal.
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare(
        const IExpression* lhs,
        const IExpression* rhs,
        Uint32 flags = 0,
        Float64 epsilon = 0.0) const = 0;

    /// Compares two instances of #mi::neuraylib::IExpression_list.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IExpression_list is defined as
    /// follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Next, the list sizes are compared using \c operator<().
    /// - Next, the lists are traversed by increasing index and the names are compared using
    ///   \c strcmp().
    /// - Finally, the list elements are enumerated by increasing index and the expressions are
    ///   compared.
    ///
    /// \param lhs          The left-hand side operand for the comparison.
    /// \param rhs          The right-hand side operand for the comparison.
    /// \param flags        A bitmask of flags of type #Comparison_options.
    /// \param epsilon      Maximum difference for floating point values to consider them as equal.
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare(
        const IExpression_list* lhs,
        const IExpression_list* rhs,
        Uint32 flags = 0,
        Float64 epsilon = 0.0) const = 0;

    //@}
    /// \name Dumping of expression and expression lists
    //@{

    /// Returns a textual representation of an expression.
    ///
    /// The parameter \p depth is only relevant for constants, where the argument is passed to
    /// #mi::neuraylib::IValue_factory::dump().
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IExpression* expr, const char* name, Size depth = 0) const = 0;

    /// Returns a textual representation of an expression list.
    ///
    /// The representation of the expression list will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IExpression_list* list, const char* name, Size depth = 0) const = 0;

    //@}
    /// \name Dumping of annotations, annotation blocks, and annotation lists
    //@{

    /// Returns a textual representation of an annotation.
    ///
    /// The representation of the annotation will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IAnnotation* annotation, const char* name, Size depth = 0) const = 0;

    /// Returns a textual representation of an annotation block.
    ///
    /// The representation of the annotation block will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IAnnotation_block* block, const char* name, Size depth = 0) const = 0;

    /// Returns a textual representation of an annotation list.
    ///
    /// The representation of the annotation list will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump(
        const IAnnotation_list* list, const char* name, Size depth = 0) const = 0;

    //@}
    /// \name Miscellaneous methods
    //@{

    /// Returns an expression which casts the source expression to a target type.
    ///
    /// This is a convenience function that creates an instance of the cast operator with the
    /// necessary arguments, stores it in the database, and returns a call expressions for the just
    /// created function. Note that the cast operator requires that the type of the source
    /// expression and the target type are compatible (see
    /// #mi::neuraylib::IType_factory::is_compatible()).
    ///
    /// \see \ref mi_neuray_mdl_cast_operator
    ///
    /// \param src_expr       The expression whose type is supposed to be cast.
    /// \param target_type    The result type of the cast. Note that the inserted cast operator
    ///                       acts on types without qualifiers, i.e., modifiers on \p target_type
    ///                       are ignored.
    /// \param cast_db_name   This name is used when storing the instance of the cast operator
    ///                       in the database. If the name is already taken by another DB element,
    ///                       this string will be used as the base for generating a unique name. If
    ///                       \c NULL, a unique name is generated.
    /// \param force_cast     This flag has only an effect if the type of the source expression is
    ///                       identical to the target type. If \c true, then the cast
    ///                       expression will still be created. If \c false, then \p src_expr itself
    ///                       will be returned.
    /// \param direct_call    Indicates whether to create a direct call expression or an (indirect)
    ///                       call expression.
    /// \param errors         An optional pointer to an #mi::Sint32 to which an error code will be
    ///                       written. The error codes have the following meaning:
    ///                       -  0: Success.
    ///                       - -1: Invalid parameters (\c NULL pointer).
    ///                       - -2: The type of \p src_expr cannot be cast to \p target_type.
    /// \return               The resulting expression or \c NULL in case of failure.
    virtual IExpression* create_cast(
        IExpression* src_expr,
        const IType* target_type,
        const char* cast_db_name,
        bool force_cast,
        bool direct_call,
        Sint32* errors = 0) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_15_0
    inline IExpression* create_cast(
        IExpression* src_expr,
        const IType* target_type,
        const char* cast_db_name,
        bool force_cast,
        Sint32* errors = 0) const
    {
        return create_cast(
            src_expr, target_type, cast_db_name, force_cast, false, errors);
    }
#endif // MI_NEURAYLIB_DEPRECATED_15_0

    /// Returns an expression which casts the source expression to a target type.
    ///
    /// This is a convenience function that creates an instance of the decl_cast operator with the
    /// necessary arguments, stores it in the database, and returns a call expressions for the just
    /// created function. Note that the decl_cast operator requires that the type of the source
    /// expression and the target type are struct types from the same struct category (see
    /// #mi::neuraylib::IType_factory::from_same_struct_category()).
    ///
    /// \see \ref mi_neuray_mdl_decl_cast_operator
    ///
    /// \param src_expr       The expression whose type is supposed to be cast.
    /// \param target_type    The result type of the cast. Note that the inserted decl_cast operator
    ///                       acts on types without qualifiers, i.e., modifiers on \p target_type
    ///                       are ignored.
    /// \param cast_db_name   This name is used when storing the instance of the decl_cast operator
    ///                       in the database. If the name is already taken by another DB element,
    ///                       this string will be used as the base for generating a unique name. If
    ///                       \c NULL, a unique name is generated.
    /// \param force_cast     This flag has only an effect if the type of the source expression is
    ///                       identical to the target type. If \c true, then the decl_cast
    ///                       expression will still be created. If \c false, then \p src_expr itself
    ///                       will be returned.
    /// \param direct_call    Indicates whether to create a direct call expression or an (indirect)
    ///                       call expression.
    /// \param errors         An optional pointer to an #mi::Sint32 to which an error code will be
    ///                       written. The error codes have the following meaning:
    ///                       -  0: Success.
    ///                       - -1: Invalid parameters (\c NULL pointer).
    ///                       - -2: The type of \p src_expr cannot be cast to \p target_type.
    /// \return               The resulting expression or \c NULL in case of failure.
    virtual IExpression* create_decl_cast(
        IExpression* src_expr,
        const IType_struct* target_type,
        const char* cast_db_name,
        bool force_cast,
        bool direct_call,
        Sint32* errors = 0) const = 0;

    //@}
};

mi_static_assert( sizeof( IExpression_factory::Comparison_options) == sizeof( mi::Uint32));

/**@}*/ // end group mi_neuray_mdl_types

}  // namespace neuraylib

}  // namespace mi

#endif // MI_NEURAYLIB_IEXPRESSION_H
