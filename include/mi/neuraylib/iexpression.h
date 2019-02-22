/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

namespace mi {

class IString;

namespace neuraylib {

class IExpression_list;

/** \addtogroup mi_neuray_mdl_types
@{
*/

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
/// Constant expressions appear as defaults of material or function definitions, as arguments of
/// material instances or function calls, as arguments of annotations, and in fields and temporaries
/// of compiled materials.
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
/// representing the actual call. See also #mi::neuraylib::IExpression_direct_call for direct call
/// expressions.
///
/// Indirect call expressions appear as defaults of material or function definitions and as
/// arguments of material instances or function calls.
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
    virtual Sint32 set_call( const char* name) = 0;
};

/// A parameter reference expression.
///
/// Parameter reference expressions appear as defaults of material or function definitions.
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
/// expressions, see #mi::neuraylib::IExpression_call).
///
/// Direct call expressions appear in fields and temporaries of compiled materials.
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
/// Temporary reference expressions appear in fields and temporaries of compiled materials.
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
    virtual const IExpression* get_expression( Size index) const = 0;

    /// Returns the expression for \p index, or \c NULL if there is no such expression.
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
    virtual const IExpression* get_expression( const char* name) const = 0;

    /// Returns the expression for \p name, or \c NULL if there is no such expression.
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
    /// \return   -  0: Success.
    ///           - -1: Invalid parameter (\c NULL pointer).
    ///           - -2: \p index is out of bounds.
    virtual Sint32 set_expression( Size index, const IExpression* expression) = 0;

    /// Sets an expression identified by name.
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

/// An annotation is similar to a direct call expression, but without corresponding definition
/// and return type.
///
/// Annotations can be created with #mi::neuraylib::IExpression_factory::create_annotation().
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
class IExpression_factory : public
    mi::base::Interface_declare<0x9fd3b2d4,0xb5b8,0x4ccd,0x9b,0x5f,0x7b,0xd9,0x9d,0xeb,0x62,0x64>
{
public:
    /// Returns the value factory associated with this expression factory.
    virtual IValue_factory* get_value_factory() const = 0;

    /// Creates a constant.
    ///
    /// \param value        The value of the constant.
    /// \return             The created constant.
    virtual IExpression_constant* create_constant( IValue* value) const = 0;

    /// Creates a call.
    ///
    /// \param name         The DB name of the referenced function call or material instance.
    /// \return             The created call.
    virtual IExpression_call* create_call( const char* name) const = 0;

    /// Creates a parameter reference.
    ///
    /// \param type         The type of the parameter.
    /// \param index        The index of the parameter.
    /// \return             The created parameter reference.
    virtual IExpression_parameter* create_parameter( const IType* type, Size index) const = 0;

    /// Creates a new expression list.
    virtual IExpression_list* create_expression_list() const = 0;

    /// Creates a new annotation.
    ///
    /// Returns \c NULL if one of the arguments is not a constant expression.
    virtual IAnnotation* create_annotation(
        const char* name, const IExpression_list* arguments) const = 0;

    /// Creates a new annotation block.
    virtual IAnnotation_block* create_annotation_block() const = 0;

    /// Creates a new annotation list.
    virtual IAnnotation_list* create_annotation_list() const = 0;

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

    /// Compares two instances of #mi::neuraylib::IExpression.
    ///
    /// The comparison operator for instances of #mi::neuraylib::IExpression is defined as follows:
    /// - If \p lhs or \p rhs is \c NULL, the result is the lexicographic comparison of
    ///   the pointer addresses themselves.
    /// - Otherwise, the types of \p lhs and \p rhs are compared. If they are different, the result
    ///   is determined by that comparison.
    /// - Next, the kind of the expressions are compared. If they are different, the result is
    ///   determined by \c operator< on the #mi::neuraylib::IExpression::Kind values.
    /// - Finally, the expressions are compared as follows:
    ///   - For constants the results is defined by comparing their values.
    ///   - For calls the result is defined by \c strcmp() on the names of the referenced DB
    ///     elements.
    ///   - For parameter and temporary references, the results is defined by \c operator<() on the
    ///     indices.
    ///
    /// \param lhs          The left-hand side operand for the comparison.
    /// \param rhs          The right-hand side operand for the comparison.
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IExpression* lhs, const IExpression* rhs) const = 0;

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
    /// \return             -1 if \c lhs < \c rhs, 0 if \c lhs == \c rhs, and +1 if \c lhs > \c rhs.
    virtual Sint32 compare( const IExpression_list* lhs, const IExpression_list* rhs) const = 0;

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
    virtual const IString* dump( const IAnnotation_block* block, const char* name, Size depth = 0)
        const = 0;

    /// Returns a textual representation of an annotation list.
    ///
    /// The representation of the annotation list will contain line breaks. Subsequent lines have a
    /// suitable indentation. The assumed indentation level of the first line is specified by \p
    /// depth.
    ///
    /// \note The exact format of the textual representation is unspecified and might change in
    ///       future releases. The textual representation is primarily meant as a debugging aid. Do
    ///       \em not base application logic on it.
    virtual const IString* dump( const IAnnotation_list* list, const char* name, Size depth = 0)
        const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

}  // namespace neuraylib

}  // namespace mi

#endif // MI_NEURAYLIB_IEXPRESSION_H
