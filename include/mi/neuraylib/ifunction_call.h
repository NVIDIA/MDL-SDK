/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief      Scene element Function_call

#ifndef MI_NEURAYLIB_IFUNCTION_CALL_H
#define MI_NEURAYLIB_IFUNCTION_CALL_H

#include <cstring>

#include <mi/neuraylib/iexpression.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/iscene_element.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_mdl_elements
@{
*/
class IMdl_execution_context;

/// This interface represents a function call.
///
/// A function call is an instance of a formal function definition, with a fixed set of arguments
/// (possibly the defaults of the function definition). Function calls can be created from
/// function definitions using #mi::neuraylib::IFunction_definition::create_function_call().
///
/// \see #mi::neuraylib::IFunction_definition, #mi::neuraylib::Argument_editor
class IFunction_call : public
    mi::base::Interface_declare<0x67b29f5f,0xdbc1,0x4300,0x94,0x53,0x3e,0x2d,0x28,0x29,0xc1,0xe9,
                                neuraylib::IScene_element>
{
public:
    /// Returns the DB name of the corresponding function definition.
    ///
    /// The type of the function definition is #mi::neuraylib::IFunction_definition.
    ///
    /// \note The DB name of the function definition is different from its MDL name (see
    ///       #get_mdl_function_definition()).
    virtual const char* get_function_definition() const = 0;

    /// Returns the MDL name of the corresponding function definition.
    ///
    /// \note The MDL name of the function definition is different from the name of the DB element
    ///       (see #get_function_definition()).
    virtual const char* get_mdl_function_definition() const = 0;

    /// Indicates whether this call is an instance of the array constructor.
    ///
    /// \see \ref mi_neuray_mdl_arrays
    inline bool is_array_constructor() const
    { return strcmp( get_mdl_function_definition(), "T[](...)") == 0; }

    /// Returns the return type of the corresponding function definition.
    virtual const IType* get_return_type() const = 0;

    /// Returns the number of parameters.
    virtual Size get_parameter_count() const = 0;

    /// Returns the name of the parameter at \p index.
    ///
    /// \param index        The index of the parameter.
    /// \return             The name of the parameter, or \c NULL if \p index is out of range.
    virtual const char* get_parameter_name( Size index) const = 0;

    /// Returns the index position of a parameter.
    ///
    /// \param name         The name of the parameter.
    /// \return             The index of the parameter, or -1 if \p name is invalid.
    virtual Size get_parameter_index( const char* name) const = 0;

    /// Returns the types of all parameters.
    virtual const IType_list* get_parameter_types() const = 0;

    /// Returns all arguments.
    virtual const IExpression_list* get_arguments() const = 0;

    /// Sets multiple arguments.
    ///
    /// \param arguments    The arguments. Note that the expressions are copied. This copy operation
    ///                     is a shallow copy, e.g., DB elements referenced in call expressions are
    ///                     \em not copied.
    /// \return
    ///                     -  0: Success.
    ///                     - -1: Invalid parameters (\c NULL pointer).
    ///                     - -2: One of the parameters in \p arguments does not exist.
    ///                     - -3: One of the argument types does not match the corresponding
    ///                           parameter type.
    ///                     - -4: The function call is immutable (because it appears in a default of
    ///                           a function or material definition).
    ///                     - -5: One of the parameter types is uniform, but the corresponding
    ///                           argument type is varying.
    ///                     - -6: One of the arguments is not a constant nor a call.
    ///                     - -7: One of the arguments contains references to DB elements in a scope
    ///                           that is more private scope than the scope of this material
    ///                           instance.
    ///                     - -8: One of the parameter types is uniform, but the corresponding
    ///                           argument is a call expression and the return type of the
    ///                           called function definition is effectively varying since the
    ///                           function definition itself is varying.
    virtual Sint32 set_arguments( const IExpression_list* arguments) = 0;

    /// Sets the argument at \p index.
    ///
    /// \param index        The index of the argument.
    /// \param argument     The argument. Note that the expression is copied. This copy operation
    ///                     is a shallow copy, e.g., DB elements referenced in call expressions are
    ///                     \em not copied.
    /// \return
    ///                     -  0: Success.
    ///                     - -1: Invalid parameters (\c NULL pointer).
    ///                     - -2: Parameter \p index does not exist.
    ///                     - -3: The argument type does not match the parameter type.
    ///                     - -4: The function call is immutable (because it appears in a default of
    ///                           a function or material definition).
    ///                     - -5: The parameter type is uniform, but the argument type is varying.
    ///                     - -6: The argument expression is not a constant nor a call.
    ///                     - -7: The argument contains references to DB elements in a scope that is
    ///                           more private scope than the scope of this material instance.
    ///                     - -8: The parameter types is uniform, but the argument is a call
    ///                           expression and the return type of the called function definition
    ///                           is effectively varying since the function definition itself is
    ///                           varying.
    virtual Sint32 set_argument( Size index, const IExpression* argument) = 0;

    /// Sets an argument identified by name.
    ///
    /// \param name         The name of the parameter.
    /// \param argument     The argument. Note that the expression is copied. This copy operation
    ///                     is a shallow copy, e.g., DB elements referenced in call expressions are
    ///                     \em not copied.
    /// \return
    ///                     -  0: Success.
    ///                     - -1: Invalid parameters (\c NULL pointer).
    ///                     - -2: Parameter \p name does not exist.
    ///                     - -3: The argument type does not match the parameter type.
    ///                     - -4: The function call is immutable (because it appears in a default of
    ///                           a function or material definition).
    ///                     - -5: The parameter type is uniform, but the argument type is varying.
    ///                     - -6: The argument expression is not a constant nor a call.
    ///                     - -7: The argument contains references to DB elements in a scope that is
    ///                           more private scope than the scope of this material instance.
    ///                     - -8: The parameter types is uniform, but the argument is a call
    ///                           expression and the return type of the called function definition
    ///                           is effectively varying since the function definition itself is
    ///                           varying.
    virtual Sint32 set_argument( const char* name, const IExpression* argument) = 0;

    /// Indicates, if this function call acts as a default argument of a material or function
    /// definition.
    ///
    /// Defaults are immutable, their arguments cannot be changed and they cannot be used
    /// in call expressions.
    ///
    /// \return true, if this function call is a default, false otherwise.
    virtual bool is_default() const = 0;

    /// Returns \c true if this function call and all its arguments point to valid
    /// material and function definitions, \c false otherwise.
    ///
    /// Material and function definitions can become invalid due to a module reload.
    ///
    /// \see #mi::neuraylib::IModule::reload(), #mi::neuraylib::IMaterial_instance::repair()
    ///
    /// \param context  Execution context that can be queried for error messages
    ///                 after the operation has finished. Can be \c NULL.
    /// \return 
    ///      - \c true:  The instance is valid.
    ///      - \c false: The instance is invalid.
    virtual bool is_valid(IMdl_execution_context* context) const = 0;

    /// Attempts to repair an invalid function call.
    ///
    /// \param flags    Repair options, see #mi::neuraylib::Mdl_repair_options.
    /// \param context  Execution context that can be queried for error messages
    ///                 after the operation has finished. Can be \c NULL.
    /// \return
    ///     -   0:   Success.
    ///     -  -1:   Repair failed. Check the \c context for details.
    virtual Sint32 repair(
        Uint32 flags,
        IMdl_execution_context* context) = 0;
};

/*@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IFUNCTION_CALL
