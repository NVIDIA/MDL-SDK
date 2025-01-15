/***************************************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief   Utility class for MDL material instances and function calls.

#ifndef MI_NEURAYLIB_ARGUMENT_EDITOR_H
#define MI_NEURAYLIB_ARGUMENT_EDITOR_H

#include <mi/base/handle.h>
#include <mi/neuraylib/assert.h>
#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/imdl_evaluator_api.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/itype.h>
#include <mi/neuraylib/ivalue.h>

#include <string>

namespace mi {

namespace neuraylib {

class IMdl_execution_context;

/** \addtogroup mi_neuray_mdl_elements
@{
*/

/// A wrapper around the interface for MDL material instances and function calls.
///
/// The purpose of the MDL argument editor is to simplify working with MDL material instances and
/// function calls. The key benefit is that it wraps API call sequences occurring in typical tasks
/// into one single method call, e.g., changing arguments (as long as their type is not too
/// complex): Typically this requires at least seven API calls (even more in case of arrays or if
/// you do not use #mi::neuraylib::set_value()). The argument editor offers a single method to
/// support this task.
///
/// Many methods are overload in the first parameter to support argument indices as well as argument
/// names. See #set_value(Size,const T&) and #set_value(const char*,const T&) for an overview of
/// the various overloads of \p %set_value(). See #get_value(Size,T&)const and
/// #get_value(const char*,T&)const  for an overview of the various overloads of \p %get_value().
///
/// \note The index-based overloads are faster than the name-based overloads and should be
/// preferred if the index/indices are known.
///
/// \note The argument editor does not expose the full functionality of the underlying interface,
/// but provides access to it via #get_scene_element().
///
/// See #mi::neuraylib::IFunction_call for the underlying interface. See also
/// #mi::neuraylib::Definition_wrapper for a similar wrapper for MDL material and function
/// definitions.
class Argument_editor
{
public:

    /// \name General methods
    //@{

    /// Constructs an MDL argument editor for a fixed material instance or function call.
    ///
    /// \param transaction       The transaction to be used.
    /// \param name              The name of the wrapped material instance or function call.
    /// \param mdl_factory       A pointer to the API component #mi::neuraylib::IMdl_factory.
    ///                          Needed by all mutable methods, can be \c NULL if only const
    ///                          methods are used.
    /// \param intent_to_edit    For best performance, the parameter should be set to \c true iff
    ///                          the intention is to edit the material instance or function call.
    ///                          This parameter is for performance optimizations only; the argument
    ///                          editor will work correctly independently of the value used. The
    ///                          performance penalty for setting it incorrectly to \c true is
    ///                          usually higher than setting it incorrectly to \c false. If in
    ///                          doubt, use the default of \c false.
    Argument_editor(
        ITransaction* transaction,
        const char* name,
        IMdl_factory* mdl_factory,
        bool intent_to_edit = false);

    /// Indicates whether the argument editor is in a valid state.
    ///
    /// The argument editor is valid if and only if the name passed in the constructor identifies a
    /// material instance or function call. This method should be immediately called after invoking
    /// the constructor. If it returns \c false, no other methods of this class should be called.
    bool is_valid() const;

    /// Indicates whether the material instance or function call referenced by this argument editor
    /// is valid. A material instance or function call is valid if itself and all calls attached
    /// to its arguments point to a valid definition.
    ///
    /// \param context  Execution context that can be queried for error messages
    ///                 after the operation has finished. Can be \c NULL.
    ///
    /// \return \c True, if the instance is valid, \c false otherwise.
    bool is_valid_instance( IMdl_execution_context* context) const;

    /// Attempts to repair an invalid material instance or function call.
    ///
    /// \param flags    Repair options, see #mi::neuraylib::Mdl_repair_options.
    /// \param context  Execution context that can be queried for error messages
    ///                 after the operation has finished. Can be \c NULL.
    /// \return
    ///     -   0:   Success.
    ///     -  -1:   Repair failed. Check the \c context for details.
    mi::Sint32 repair( mi::Uint32 flags, IMdl_execution_context* context);

    /// Indicates whether the argument editor acts on a material instance or on a function call.
    ///
    /// \return    #mi::neuraylib::ELEMENT_TYPE_FUNCTION_DEFINITION, or undefined if #is_valid()
    ///            returns \c false.
    Element_type get_type() const;

    /// Returns the DB name of the corresponding material or function definition.
    const char* get_definition() const;

    /// Returns the MDL name of the corresponding material or function definition.
    const char* get_mdl_definition() const;

    /// Indicates whether the argument editor acts on a function call that is an instance of the
    /// array constructor.
    ///
    /// \see \ref mi_neuray_mdl_arrays
    bool is_array_constructor() const;

    /// Indicates whether the corresponding material or function definition is declarative.
    bool is_declarative() const;

    /// Indicates whether the argument editor acts on a material instance.
    bool is_material() const;

    /// Returns the return type.
    const IType* get_return_type() const;

    /// Returns the number of parameters.
    Size get_parameter_count() const;

    /// Returns the name of the parameter at \p index.
    ///
    /// \param index    The index of the parameter.
    /// \return         The name of the parameter, or \c NULL if \p index is out of range.
    const char* get_parameter_name( Size index) const;

    /// Returns the index position of a parameter.
    ///
    /// \param name     The name of the parameter.
    /// \return         The index of the parameter, or -1 if \p name is invalid.
    Size get_parameter_index( const char* name) const;

    /// Returns the types of all parameters.
    const IType_list* get_parameter_types() const;

    /// Checks the \c enable_if condition of the given parameter.
    ///
    /// \param index      The index of the parameter.
    /// \param evaluator  A pointer to the API component #mi::neuraylib::IMdl_evaluator_api.
    /// \return           \c false if the \c enable_if condition of this parameter evaluated to
    ///                   \c false, \c true otherwise
    bool is_parameter_enabled( Size index, IMdl_evaluator_api* evaluator) const;

    /// Returns all arguments.
    const IExpression_list* get_arguments() const;

    /// Returns the expression kind of an argument.
    ///
    /// This method should be used to figure out whether
    /// #mi::neuraylib::Argument_editor::get_value() or #mi::neuraylib::Argument_editor::get_call()
    /// should be used for reading an argument.
    IExpression::Kind get_argument_kind( Size parameter_index) const;

    /// Returns the expression kind of an argument.
    ///
    /// This method should be used to figure out whether
    /// #mi::neuraylib::Argument_editor::get_value() or #mi::neuraylib::Argument_editor::get_call()
    /// should be used for reading an argument.
    IExpression::Kind get_argument_kind( const char* parameter_name) const;

    //@}
    /// \name Methods related to resetting of arguments
    //@{

    /// Resets the argument at \p index.
    ///
    /// If the definition has a default for this parameter (and it does not violate a
    /// potential uniform requirement), then a clone of it is used as new argument. Otherwise, a
    /// constant expression is created, observing range annotations if present (see the overload of
    /// #mi::neuraylib::IValue_factory::create() with two arguments).
    ///
    /// \param index        The index of the argument.
    /// \return
    ///                     -   0: Success.
    ///                     -  -2: Parameter \p index does not exist.
    ///                     -  -4: The function call or material instance is immutable (because it
    ///                            appears in a default of a function or material definition).
    ///                     -  -9: The function call or material instance is not valid (see
    ///                            #is_valid()).
    Sint32 reset_argument( Size index);

    /// Sets an argument identified by name to its default.
    ///
    /// If the definition has a default for this parameter (and it does not violate a
    /// potential uniform requirement), then a clone of it is used as new argument. Otherwise, a
    /// constant expression is created, observing range annotations if present (see the overload of
    /// #mi::neuraylib::IValue_factory::create() with two arguments).
    ///
    /// \param name         The name of the parameter.
    /// \return
    ///                     -   0: Success.
    ///                     -  -1: Invalid parameters (\c NULL pointer).
    ///                     -  -2: Parameter \p name does not exist.
    ///                     -  -4: The function call or material instance is immutable (because it
    ///                            appears in a default of a function or material definition).
    ///                     -  -9: The function call or material instance is not valid (see
    ///                            #is_valid()).
    Sint32 reset_argument( const char* name);

    //@}
    /// \name Methods related to constant expressions
    //@{

    /// Returns an argument (values of constants only, no calls).
    ///
    /// This method supports all atomic MDL types with the corresponding C-type as type of \p value
    /// (and \c int for #mi::neuraylib::IValue_enum). Vectors and matrices are supported if \p value
    /// is of type #mi::math::Vector and #mi::math::Matrix, respectively. For arrays, see
    /// #get_value(Size,T*,Size)const. For components of compounds, see
    /// #get_value(Size,Size,T&)const and #get_value(Size,const char*,T&)const.
    ///
    /// It is not possible to read entire structs with a single call (in general there is no
    /// corresponding C++ class, and absence of introspection machinery). Struct fields need to
    /// be read one by one.
    ///
    /// There is no support for inner-most components of multi-dimensional compounds (arrays of
    /// compounds or structs of compounds) -- this would require additional overloads accepting two
    /// or more component indices and/or field names.
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 get_value( Size parameter_index, T& value) const;

    /// Returns an argument (values of constants only, no calls).
    ///
    /// This method supports all atomic MDL types with the corresponding C-type as type of \p value
    /// (and \c int for #mi::neuraylib::IValue_enum). Vectors and matrices are supported if \p value
    /// is of type #mi::math::Vector and #mi::math::Matrix, respectively. For arrays, see
    /// #get_value(const char*,T*,Size)const. For components of compounds, see
    /// #get_value(const char*,Size,T&)const and #get_value(const char*,const char*,T&)const.
    ///
    /// It is not possible to read entire structs with a single call (in general there is no
    /// corresponding C++ class, and absence of introspection machinery). Struct fields need to
    /// be read one by one.
    ///
    /// There is no support for inner-most components of multi-dimensional compounds (arrays of
    /// compounds or structs of compounds) -- this would require additional overloads accepting two
    /// or more component indices and/or field names.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template <class T>
    Sint32 get_value( const char* parameter_name, T& value) const;

    //@}
    /// \name Methods related to constant expressions (arrays)
    //@{

    /// Returns an array argument (values of constants only, no calls).
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param[out] value       The current value of the specified argument.
    /// \param n                The size of the C array (needs to match the size of the argument
    ///                         identified by \p parameter_index).
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T or the array size \p n.
    template<class T>
    Sint32 get_value( Size parameter_index, T* value, Size n) const;

    /// Returns an array argument (values of constants only, no calls).
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param[out] value       The current value of the specified argument.
    /// \param n                The size of the C array (needs to match the size of the argument
    ///                         identified by \p parameter_name).
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T or the array size \p n.
    template <class T>
    Sint32 get_value( const char* parameter_name, T* value, Size n) const;

    //@}
    /// \name Methods related to constant expressions (components of compounds)
    //@{

    /// Returns a component of a compound argument (values of constants only, no calls).
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param component_index  The index of the component in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -3: \p component_index is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 get_value( Size parameter_index, Size component_index, T& value) const;

    /// Returns a component of a compound argument (values of constants only, no calls).
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param component_index  The index of the component in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -3: \p component_index is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 get_value( const char* parameter_name, Size component_index, T& value) const;

    /// Returns a field of a struct argument (values of constants only, no calls).
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param field_name       The name of the struct field in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -3: \p field_name is invalid.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 get_value( Size parameter_index, const char* field_name, T& value) const;

    /// Returns a field of a struct argument (values of constants only, no calls).
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param field_name       The name of the struct field in question.
    /// \param[out] value       The current value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -3: \p field_name is invalid.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template <class T>
    Sint32 get_value( const char* parameter_name, const char* field_name, T& value) const;

    //@}
    /// \name Methods related to constant expressions
    //@{

    /// Sets an argument.
    ///
    /// This method supports all atomic MDL types with the corresponding C-type as type of \p value
    /// (and \c int for #mi::neuraylib::IValue_enum). Vectors and matrices are supported if \p value
    /// is of type #mi::math::Vector and #mi::math::Matrix, respectively. For arrays, see
    /// #set_value(Size,const T*,Size). For components of compounds, see
    /// #set_value(Size,Size,const T&) and #set_value(Size,const char*,const T&).
    ///
    /// It is not possible to set entire structs with a single call (in general there is no
    /// corresponding C++ class, and absence of introspection machinery). Struct fields need to
    /// be set one by one.
    ///
    /// There is no support for inner-most components of multi-dimensional compounds (arrays of
    /// compounds or structs of compounds) -- this would require additional overloads accepting two
    /// or more component indices and/or field names.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 set_value( Size parameter_index, const T& value);

    /// Sets an argument.
    ///
    /// This method supports all atomic MDL types with the corresponding C-type as type of \p value
    /// (and \c int for #mi::neuraylib::IValue_enum). Vectors and matrices are supported if \p value
    /// is of type #mi::math::Vector and #mi::math::Matrix, respectively. For arrays, see
    /// #set_value(const char*,const T*,Size). For components of compounds, see
    /// #set_value(const char*,Size,const T&) and #set_value(const char*,const char*,const T&).
    ///
    /// It is not possible to set entire structs with a single call (in general there is no
    /// corresponding C++ class, and absence of introspection machinery). Struct fields need to
    /// be set one by one.
    ///
    /// There is no support for inner-most components of multi-dimensional compounds (arrays of
    /// compounds or structs of compounds) -- this would require additional overloads accepting two
    /// or more component indices and/or field names.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template <class T>
    Sint32 set_value( const char* parameter_name, const T& value);

    //@}
    /// \name Methods related to constant expressions (arrays)
    //@{

    /// Sets an array argument (values of constants only, no calls).
    ///
    /// If the argument is a deferred-sized array, then its size is adjusted according to the
    /// parameter \p n. If the argument is an immediate-sized array, then its size needs to match
    /// the parameter \p n.
    ///
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param value            The new value of the specified argument.
    /// \param n                The size of the C array (needs to match the size of the argument
    ///                         identified by \p parameter_index in case of immediate-sized arrays).
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T or the array size \p n.
    template<class T>
    Sint32 set_value( Size parameter_index, const T* value, Size n);

    /// Sets an array argument (values of constants only, no calls).
    ///
    /// If the argument is a deferred-sized array, then its size is adjusted according to the
    /// parameter \p n. If the argument is an immediate-sized array, then its size needs to match
    /// the parameter \p n.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param value            The new value of the specified argument.
    /// \param n                The size of the C array (needs to match the size of the argument
    ///                         identified by \p parameter_index in case of immediate-sized arrays).
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T or the array size \p n.
    template <class T>
    Sint32 set_value( const char* parameter_name, const T* value, Size n);

    //@}
    /// \name Methods related to constant expressions (components of compounds)
    //@{

    /// Sets a component of a compound argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// This method does not change the size of deferred-sized arrays. You need to call explicitly
    /// #mi::neuraylib::Argument_editor::set_array_size() for that.
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param component_index  The index of the component in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -3: \p component_index is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 set_value( Size parameter_index, Size component_index, const T& value);

    /// Sets a component of a compound argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// This method does not change the size of deferred-sized arrays. You need to call explicitly
    /// #mi::neuraylib::Argument_editor::set_array_size() for that.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param component_index  The index of the component in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -3: \p component_index is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template <class T>
    Sint32 set_value( const char* parameter_name, Size component_index, const T& value);

    /// Sets a field of a struct argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param field_name       The name of the struct_field in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -3: \p field_name is invalid.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template<class T>
    Sint32 set_value( Size parameter_index, const char* field_name, const T& value);

    /// Sets a field of a struct argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param field_name       The name of the struct_field in question.
    /// \param value            The new value of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -3: \p field_name is invalid.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The type of the argument does not match the template
    ///                               parameter \p T.
    template <class T>
    Sint32 set_value( const char* parameter_name, const char* field_name, const T& value);

    //@}
    /// \name Methods related to constant expressions (arrays)
    //@{

    /// Returns the length of an array argument.
    ///
    /// If a literal \c 0 is passed for \p argument_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Uint32.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param[out] size        The current length of the array of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The argument is not an array.
    Sint32 get_array_length( Uint32 parameter_index, Size& size) const;

    /// Returns the length of an array argument.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param[out] size        The current length of the array of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -4: The argument is not a constant.
    ///                         - -5: The argument is not an array.
    Sint32 get_array_length( const char* parameter_name, Size& size) const;

    /// Sets the length of a deferred-sized array argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// If a literal \c 0 is passed for \p argument_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Uint32.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param size             The new length of the array of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_index is out of range.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The argument is not a deferred-sized array.
    Sint32 set_array_size( Uint32 parameter_index, Size size);

    /// Sets the length of a deferred-sized array argument.
    ///
    /// If the current argument is a call expression, it will be replaced by a constant expression.
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param size             The new length of the array of the specified argument.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The argument is not a deferred-sized array.
    Sint32 set_array_size( const char* parameter_name, Size size);

    //@}
    /// \name Methods related to call expressions
    //@{

    /// Returns an argument (call expressions only).
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \return                 The name of the call, or \c NULL if \p parameter_index is out of
    ///                         bounds or the argument expression is not a call.
    const char* get_call( Size parameter_index) const;

    /// Returns an argument (call expressions only).
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \return                 The name of the call, or \c NULL if \p parameter_name is invalid
    ///                         or the argument expression is not a call.
    const char* get_call( const char* parameter_name) const;

    /// Sets an argument (call expressions only).
    ///
    /// If a literal \c 0 is passed for \p parameter_index, the call is ambiguous. You need to
    /// explicitly cast the value to #mi::Size.
    ///
    /// \param parameter_index  The index of the argument in question.
    /// \param call_name        The name of the call to set.
    /// \return
    ///                         -   0: Success.
    ///                         -  -1: #is_valid() returns \c false.
    ///                         -  -2: \p parameter_index is out of range.
    ///                         -  -3: The return type of \p call_name does not match the argument
    ///                                type.
    ///                         -  -4: The material instance or function call is immutable.
    ///                         -  -5: The parameter is uniform, but the return type of the
    ///                                call is varying.
    ///                         -  -6: \p call_name is invalid.
    ///                         -  -8: The parameter is uniform, but the argument is a call
    ///                                expression and the return type of the called function
    ///                                definition is effectively varying since the function
    ///                                definition itself is varying.
    ///                         - -10: The definition is non-declarative, but at least one of the
    ///                                arguments is a declarative call.
    Sint32 set_call( Size parameter_index, const char* call_name);

    /// Sets an argument (call expressions only).
    ///
    /// \param parameter_name   The name of the argument in question.
    /// \param call_name        The name of the call to set.
    /// \return
    ///                         -  0: Success.
    ///                         - -1: #is_valid() returns \c false.
    ///                         - -2: \p parameter_name is invalid.
    ///                         - -3: The return type of \p call_name does not match the argument
    ///                               type.
    ///                         - -4: The material instance or function call is immutable.
    ///                         - -5: The parameter is uniform, but the return type of the
    ///                               call is varying.
    ///                         - -6: \p call_name is invalid.
    Sint32 set_call( const char* parameter_name, const char* call_name);

    //@}
    /// \name Methods related to member access.
    //@{

    /// Get the transaction.
    ITransaction* get_transaction() const;

    /// Get the MDL factory.
    IMdl_factory* get_mdl_factory() const;

    /// Get the value factory.
    IValue_factory* get_value_factory() const;

    /// Get the expression factory.
    IExpression_factory* get_expression_factory() const;

    /// Get the MDL function call or material instance.
    const IFunction_call* get_scene_element() const;

    /// Get the MDL function call or material instance.
    IFunction_call* get_scene_element();

    /// Get the element type.
    Element_type get_element_type() const;

    /// Get the DB name of the MDL function call or material instance.
    const std::string& get_name() const;

    //@}

private:
    void promote_to_edit_if_needed();

    base::Handle<ITransaction> m_transaction;
    base::Handle<IMdl_factory> m_mdl_factory;
    base::Handle<IValue_factory> m_value_factory;
    base::Handle<IExpression_factory> m_expression_factory;
    base::Handle<const IFunction_call> m_access;
    base::Handle<const IFunction_call> m_old_access;
    base::Handle<IFunction_call> m_edit;
    Element_type m_type;
    std::string m_name;
};

/**@}*/ // end group mi_neuray_mdl_elements

inline Argument_editor::Argument_editor(
    ITransaction* transaction, const char* name, IMdl_factory* mdl_factory, bool intent_to_edit)
{
    mi_neuray_assert( transaction);
    mi_neuray_assert( name);

    m_transaction = make_handle_dup( transaction);
    m_name = name;
    m_mdl_factory = make_handle_dup( mdl_factory);
    m_value_factory
        = m_mdl_factory ? m_mdl_factory->create_value_factory( m_transaction.get()) : 0;
    m_expression_factory
        = m_mdl_factory ? m_mdl_factory->create_expression_factory( m_transaction.get()) : 0;

    if( intent_to_edit) {
        m_edit = transaction->edit<IFunction_call>( name);
        m_access = m_edit;
        m_old_access = m_edit;
        m_type = m_access ? m_access->get_element_type() : static_cast<Element_type>( 0);
    } else {
        m_access = transaction->access<IFunction_call>( name);
        m_type = m_access ? m_access->get_element_type() : static_cast<Element_type>( 0);
    }
}

inline bool Argument_editor::is_valid() const
{
    return m_access.is_valid_interface();
}

inline bool Argument_editor::is_valid_instance( IMdl_execution_context* context) const
{
    if( !is_valid())
        return false;

    return m_access->is_valid( context);
}

inline mi::Sint32 Argument_editor::repair( mi::Uint32 flags, IMdl_execution_context* context)
{
    if( !is_valid())
        return false;

    promote_to_edit_if_needed();

    return m_edit->repair( flags, context);
}

inline Element_type Argument_editor::get_type() const
{
    return m_type;
}

inline const char* Argument_editor::get_definition() const
{
    if( !is_valid())
        return 0;

    return m_access->get_function_definition();
}

inline const char* Argument_editor::get_mdl_definition() const
{
    if( !is_valid())
        return 0;

    return m_access->get_mdl_function_definition();
}

inline bool Argument_editor::is_array_constructor() const
{
    if( !is_valid())
        return false;

    return m_access->is_array_constructor();
}

inline bool Argument_editor::is_declarative() const
{
    if( !is_valid())
        return false;

    return m_access->is_declarative();
}

inline bool Argument_editor::is_material() const
{
    if( !is_valid())
        return false;

    return m_access->is_material();
}

inline Size Argument_editor::get_parameter_count() const
{
    if( !is_valid())
        return 0;

    return m_access->get_parameter_count();
}

inline const char* Argument_editor::get_parameter_name( Size parameter_index) const
{
    if( !is_valid())
        return 0;

    return m_access->get_parameter_name( parameter_index);
}

inline Size Argument_editor::get_parameter_index( const char* name) const
{
    if( !is_valid())
        return 0;

    return m_access->get_parameter_index( name);
}

inline const IType* Argument_editor::get_return_type() const
{
    if( !is_valid())
        return 0;

    return m_access->get_return_type();
}

inline const IType_list* Argument_editor::get_parameter_types() const
{
    if( !is_valid())
        return 0;

    return m_access->get_parameter_types();
}

inline bool Argument_editor::is_parameter_enabled( Size index, IMdl_evaluator_api* evaluator) const
{
    if( !evaluator)
        return true;

    if( !is_valid())
        return true;

    base::Handle<const IValue_bool> b( evaluator->is_function_parameter_enabled(
        m_transaction.get(), m_value_factory.get(), m_access.get(), index, /*errors*/ 0));
    if( !b)
        return true;
    return b->get_value();
}

inline Sint32 Argument_editor::reset_argument( Size index)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    return m_edit->reset_argument( index);
}

inline Sint32 Argument_editor::reset_argument( const char* name)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    return m_edit->reset_argument( name);
}


inline const IExpression_list* Argument_editor::get_arguments() const
{
    if( !is_valid())
        return 0;

    return m_access->get_arguments();
}

inline IExpression::Kind Argument_editor::get_argument_kind( Size parameter_index) const
{
    if( !is_valid())
        return static_cast<IExpression::Kind>( 0);

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    return argument ? argument->get_kind() : static_cast<IExpression::Kind>( 0);
}

inline IExpression::Kind Argument_editor::get_argument_kind( const char* parameter_name) const
{
    if( !is_valid())
        return static_cast<IExpression::Kind>( 0);

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    return argument ? argument->get_kind() : static_cast<IExpression::Kind>( 0);
}

template <class T>
Sint32 Argument_editor::get_value( Size parameter_index, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), value);
    return result == 0 ? 0 : -5;
}

template <class T>
Sint32 Argument_editor::get_value( const char* name, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument_( arguments->get_expression( name));
    if( !argument_)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument_->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), value);
    return result == 0 ? 0 : -5;
}

template <class T>
Sint32 Argument_editor::get_value( Size parameter_index, T* value, Size n) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), value, n);
    return result == 0 ? 0 : -5;
}

template <class T>
Sint32 Argument_editor::get_value( const char* name, T* value, Size n) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument_( arguments->get_expression( name));
    if( !argument_)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument_->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), value, n);
    return result == 0 ? 0 : -5;
}

template <class T>
Sint32 Argument_editor::get_value( Size parameter_index, Size component_index, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), component_index, value);
    return result == 0 ? 0 : (result == -3 ? -3 : -5);
}

template <class T>
Sint32 Argument_editor::get_value( const char* parameter_name, Size component_index, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), component_index, value);
    return result == 0 ? 0 : (result == -3 ? -3 : -5);
}

template <class T>
Sint32 Argument_editor::get_value(
    Size parameter_index, const char* field_name, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), field_name, value);
    return result == 0 ? 0 : (result == -3 ? -3 : -5);
}

template <class T>
Sint32 Argument_editor::get_value(
    const char* parameter_name, const char* field_name, T& value) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue> argument_value( argument_constant->get_value());
    Sint32 result = neuraylib::get_value( argument_value.get(), field_name, value);
    return result == 0 ? 0 : (result == -3 ? -3 : -5);
}

template <class T>
Sint32 Argument_editor::set_value( Size parameter_index, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IType> type( argument->get_type());
    base::Handle<IValue> new_value( m_value_factory->create( type.get()));
    Sint32 result = neuraylib::set_value( new_value.get(), value);
    if( result != 0)
        return -5;
    base::Handle<IExpression> new_expression(
        m_expression_factory->create_constant( new_value.get()));
    result = m_edit->set_argument( parameter_index, new_expression.get());
    mi_neuray_assert( result == 0);
    return result;
}

template <class T>
Sint32 Argument_editor::set_value( const char* parameter_name, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    base::Handle<const IType> type( argument->get_type());
    base::Handle<IValue> new_value( m_value_factory->create( type.get()));
    Sint32 result = neuraylib::set_value( new_value.get(), value);
    if( result != 0)
        return -5;
    base::Handle<IExpression> new_expression(
        m_expression_factory->create_constant( new_value.get()));
    result = m_edit->set_argument( parameter_name, new_expression.get());
    mi_neuray_assert( result == 0);
    return result;
}

template <class T>
Sint32 Argument_editor::set_value( Size parameter_index, const T* value, Size n)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IType_array> type( argument->get_type<IType_array>());
    base::Handle<IValue_array> new_value( m_value_factory->create_array( type.get()));
    if( !new_value)
        return -5;
    if( !type->is_immediate_sized())
        new_value->set_size( n);
    Sint32 result = neuraylib::set_value( new_value.get(), value, n);
    if( result != 0)
        return -5;
    base::Handle<IExpression> new_expression(
        m_expression_factory->create_constant( new_value.get()));
    result = m_edit->set_argument( parameter_index, new_expression.get());
    mi_neuray_assert( result == 0);
    return result;
}

template <class T>
Sint32 Argument_editor::set_value( const char* parameter_name, const T* value, Size n)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    base::Handle<const IType_array> type( argument->get_type<IType_array>());
    base::Handle<IValue_array> new_value( m_value_factory->create_array( type.get()));
    if( !new_value)
        return -5;
    if( !type->is_immediate_sized())
        new_value->set_size( n);
    Sint32 result = neuraylib::set_value( new_value.get(), value, n);
    if( result != 0)
        return -5;
    base::Handle<IExpression> new_expression(
        m_expression_factory->create_constant( new_value.get()));
    result = m_edit->set_argument( parameter_name, new_expression.get());
    mi_neuray_assert( result == 0);
    return result;
}

template <class T>
Sint32 Argument_editor::set_value( Size parameter_index, Size component_index, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue> new_value( new_argument_constant->get_value());
        Sint32 result = neuraylib::set_value( new_value.get(), component_index, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        result = m_edit->set_argument( parameter_index, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue> new_value( m_value_factory->create( type.get()));
        Sint32 result = neuraylib::set_value( new_value.get(), component_index, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_index, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

template <class T>
Sint32 Argument_editor::set_value( const char* parameter_name, Size component_index, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue> new_value( new_argument_constant->get_value());
        Sint32 result = neuraylib::set_value( new_value.get(), component_index, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        result = m_edit->set_argument( parameter_name, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue> new_value( m_value_factory->create( type.get()));
        Sint32 result = neuraylib::set_value( new_value.get(), component_index, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_name, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

template <class T>
Sint32 Argument_editor::set_value(
    Size parameter_index, const char* field_name, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue> new_value( new_argument_constant->get_value());
        Sint32 result = neuraylib::set_value( new_value.get(), field_name, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        result = m_edit->set_argument( parameter_index, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue> new_value( m_value_factory->create( type.get()));
        Sint32 result = neuraylib::set_value( new_value.get(), field_name, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_index, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

template <class T>
Sint32 Argument_editor::set_value(
    const char* parameter_name, const char* field_name, const T& value)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue> new_value( new_argument_constant->get_value());
        Sint32 result = neuraylib::set_value( new_value.get(), field_name, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        result = m_edit->set_argument( parameter_name, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue> new_value( m_value_factory->create( type.get()));
        Sint32 result = neuraylib::set_value( new_value.get(), field_name, value);
        if( result != 0)
            return result == -3 ? -3 : -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_name, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

inline Sint32 Argument_editor::get_array_length( Uint32 parameter_index, Size& size) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue_array> value( argument_constant->get_value<IValue_array>());
    if( !value)
        return -5;
    size = value->get_size();
    return 0;
}

inline Sint32 Argument_editor::get_array_length( const char* parameter_name, Size& size) const
{
    if( !is_valid())
        return -1;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    base::Handle<const IExpression_constant> argument_constant(
        argument->get_interface<IExpression_constant>());
    if( !argument_constant)
        return -4;
    base::Handle<const IValue_array> value( argument_constant->get_value<IValue_array>());
    if( !value)
        return -5;
    size = value->get_size();
    return 0;
}

inline Sint32 Argument_editor::set_array_size( Uint32 parameter_index, Size size)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_index));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue_array> new_value(
            new_argument_constant->get_value<IValue_array>());
        if( !new_value)
            return -5;
        Sint32 result = new_value->set_size( size);
        if( result != 0)
            return -5;
        result = m_edit->set_argument( parameter_index, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue_array> new_value(
            m_value_factory->create<IValue_array>( type.get()));
        if( !new_value)
            return -5;
        Sint32 result = new_value->set_size( size);
        if( result != 0)
            return -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_index, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

inline Sint32 Argument_editor::set_array_size( const char* parameter_name, Size size)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    if( m_edit->is_default())
        return -4;
    base::Handle<const IExpression_list> arguments( m_edit->get_arguments());
    base::Handle<const IExpression> argument( arguments->get_expression( parameter_name));
    if( !argument)
        return -2;
    if( argument->get_kind() == IExpression::EK_CONSTANT) {
        // reuse existing constant expression
        base::Handle<IExpression> new_argument( m_expression_factory->clone( argument.get()));
        base::Handle<IExpression_constant> new_argument_constant(
            new_argument->get_interface<IExpression_constant>());
        base::Handle<IValue_array> new_value(
            new_argument_constant->get_value<IValue_array>());
        if( !new_value)
            return -5;
        Sint32 result = new_value->set_size( size);
        if( result != 0)
            return -5;
        result = m_edit->set_argument( parameter_name, new_argument.get());
        mi_neuray_assert( result == 0);
        return result;
    } else {
        // create new constant expression
        base::Handle<const IType> type( argument->get_type());
        base::Handle<IValue_array> new_value(
            m_value_factory->create<IValue_array>( type.get()));
        if( !new_value)
            return -5;
        Sint32 result = new_value->set_size( size);
        if( result != 0)
            return -5;
        base::Handle<IExpression> new_expression(
            m_expression_factory->create_constant( new_value.get()));
        result = m_edit->set_argument( parameter_name, new_expression.get());
        mi_neuray_assert( result == 0);
        return result;
    }
}

inline const char* Argument_editor::get_call( Size parameter_index) const
{
    if( !is_valid())
        return 0;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression_call> argument(
        arguments->get_expression<IExpression_call>( parameter_index));
    if( !argument)
        return 0;
    return argument->get_call();
}

inline const char* Argument_editor::get_call( const char* parameter_name) const
{
    if( !is_valid())
        return 0;

    base::Handle<const IExpression_list> arguments( m_access->get_arguments());
    base::Handle<const IExpression_call> argument(
        arguments->get_expression<IExpression_call>( parameter_name));
    if( !argument)
        return 0;
    return argument->get_call();
}

inline Sint32 Argument_editor::set_call( Size parameter_index, const char* call_name)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    base::Handle<IExpression_call> new_argument( m_expression_factory->create_call( call_name));
    if( !new_argument)
        return -6;
    return m_edit->set_argument( parameter_index, new_argument.get());
}

inline Sint32 Argument_editor::set_call( const char* parameter_name, const char* call_name)
{
    if( !is_valid())
        return -1;

    promote_to_edit_if_needed();

    base::Handle<IExpression_call> new_argument( m_expression_factory->create_call( call_name));
    if( !new_argument)
        return -6;
    return m_edit->set_argument( parameter_name, new_argument.get());
}

inline ITransaction* Argument_editor::get_transaction() const
{
    m_transaction->retain();
    return m_transaction.get();
}

inline IMdl_factory* Argument_editor::get_mdl_factory() const
{
    m_mdl_factory->retain();
    return m_mdl_factory.get();
}

inline IValue_factory* Argument_editor::get_value_factory() const
{
    m_value_factory->retain();
    return m_value_factory.get();
}

inline IExpression_factory* Argument_editor::get_expression_factory() const
{
    m_expression_factory->retain();
    return m_expression_factory.get();
}

inline const IFunction_call* Argument_editor::get_scene_element() const
{
    m_access->retain();
    return m_access.get();
}

inline IFunction_call* Argument_editor::get_scene_element() //-V659 PVS
{
    promote_to_edit_if_needed();

    m_edit->retain();
    return m_edit.get();
}

inline Element_type Argument_editor::get_element_type() const
{
    return m_type;
}

inline const std::string& Argument_editor::get_name() const
{
    return m_name;
}

inline void Argument_editor::promote_to_edit_if_needed()
{
    if( m_edit)
        return;

    m_edit = m_transaction->edit<IFunction_call>( m_name.c_str());
    mi_neuray_assert( m_edit);
    m_old_access = m_access;
    m_access = m_edit;
}

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ARGUMENT_EDITOR_H
