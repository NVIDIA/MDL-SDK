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
/// \brief      Scene element Module

#ifndef MI_NEURAYLIB_IMODULE_H
#define MI_NEURAYLIB_IMODULE_H

#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/version.h>


namespace mi {

class IArray;

namespace neuraylib {

class IAnnotation_block;
class IAnnotation_definition;
class IExpression_list;
class IMdl_execution_context;
class IType_list;
class IType_resource;
class IValue_list;
class IValue_resource;

/** \defgroup mi_neuray_mdl_elements MDL-related elements
    \ingroup mi_neuray_scene_element
*/

/** \addtogroup mi_neuray_mdl_elements

@{

MDL-related elements comprise a set of interfaces related to the Material Definition Language (MDL).

See [\ref MDLTI] for a technical introduction into the Material Definition Language and [\ref MDLLS]
for the language specification. See also \ref mi_neuray_mdl_types.

The unit of compilation in MDL is a module. Importing an MDL module creates an instance of
#mi::neuraylib::IModule in the DB. A module allows to retrieve the referenced (aka imported)
modules, as well as the exported material and function definitions. For all exported definitions,
instances of #mi::neuraylib::IMaterial_definition and #mi::neuraylib::IFunction_definition
are created in the DB accordingly. Both, material and function definitions can be instantiated.
Those instantiations are represented by the interfaces #mi::neuraylib::IMaterial_instance and
#mi::neuraylib::IFunction_call.

The DB names of all scene elements related to MDL always carry the prefix \c "mdl". This prefix is
followed by the fully qualified MDL name of the entity including two initial colons, e.g., the DB
element representing the \c df module has the DB name \c "mdl::df". Since function definitions can
be overloaded in MDL, the DB names contain the function signature, e.g., if there are two functions
\c "bool foo(int n)" and \c "bool foo(float f)" in module \c bar, the DB name of these are \c
"mdl::bar::foo(int)" and \c "mdl::bar::foo(float)" respectively.

When instantiating a material or function definition, its formal parameters are provided with actual
arguments. If the parameter has a default, the argument can be omitted in which case the default is
used. Arguments of instances can also be changed after instantiation.

\section mi_neuray_mdl_structs Structs

For each exported struct type function definitions for its constructors are created in the DB. There
is a default constructor and a so-called elemental constructor which has a parameter for each field
of the struct. The name of these constructors is the name of the struct type including the
signature.

\par Example
The MDL code
\code
export struct Foo {
    int param_int;
    float param_float = 0;
};
\endcode
in a module \c "mod_struct" creates the following function definitions:
- a default constructor named \c "mdl::mod_struct::Foo()",
- an elemental constructor named \c "mdl::mod_struct::Foo(int,float)", and
.
The elemental constructor has a parameter \c "param_int" of type #mi::neuraylib::IType_int and a
parameter \c "param_float" of type #mi::neuraylib::IType_float. Both function definitions have
the return type #mi::neuraylib::IType_struct with name \c "::mod_struct::Foo".

In addition, for each exported struct type, and for each of its fields, a function definition for
the corresponding member selection operator is created in the DB. The name of that function
definition is obtained by concatenating the name of the struct type with the name of the field with
an intervening dot, e.g., \c "foo.bar". The function definition has a single parameter \c "s" of the
struct type and the corresponding field type as return type.

\par Example
The MDL code
\code
export struct Foo {
    int param_int;
    float param_float = 0;
};
\endcode
in a module \c "mod_struct" creates the two function definitions named \c
"mdl::struct::Foo.param_int(::struct::Foo)" and \c "mdl::struct::Foo.param_float(::struct::Foo)" to
represent the member selection operators \c "Foo.param_int" and \c "Foo.param_float". The function
definitions have a single parameter \c "s" of type \c "mdl::struct::Foo" and return type
#mi::neuraylib::IType_int and #mi::neuraylib::IType_float, respectively.

\section mi_neuray_mdl_arrays Arrays

In contrast to struct types which are explicitly declared there are infinitely many array types
(considering pairs of element type and array length). Deferred-sized arrays make the situation even
more complicated. Each of these array types would its own constructor and its own index operator.
Therefore, template-like functions definitions are used in the context of arrays to satisfy this
need. See the next section for details

\section mi_neuray_mdl_template_like_functions_definitions Template-like function definitions

Usually, function definitions have a fixed set of parameter types and a fixed return type. As an
exception of this rule, there are five function definitions which rather have the character of a
template with generic parameter and/or return types.

- the array constructor (#mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR),
- the array length operator (#mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH),
- the array index operator (#mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX),
- the ternary operator (#mi::neuraylib::IFunction_definition::DS_TERNARY), and
- the cast operator (#mi::neuraylib::IFunction_definition::DS_CAST).

The MDL and DB names of these function definitions use \c "<0>" or \c "T" to indicate such a generic
parameter or return type. When querying the actual type, #mi::neuraylib::IType_int (arbitrary
choice) is returned for lack of a better alternative.

When creating function calls from such template-like function definitions, the parameter and return
types are fixed, i.e., the function call itself has concrete parameter and return types as usual,
and has no template-like behavior as the definition from which it was created. This implies that,
for example, after creation of a ternary operator on floats, you cannot set its arguments to
expressions of a different type than float (this would require creation of another function call
with the desired parameter types).

More details about the five different template-like function definitions follow.


\subsection mi_neuray_mdl_array_constructor Array constructor

Semantic: #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR \n
DB name: \c "mdl::T[](...)" \n
MDL name: \c "T[](...)"

The following requirements apply when creating function calls of the array constructor:
- the expression list for the arguments contains a non-zero number of arguments,
- all arguments must be of the same type (which is the element type of the constructed array), and
- the positional order of the arguments in the expression list is used, their names are irrelevant
  (although the expression list itself requires that they are distinct).


\subsection mi_neuray_mdl_array_length_operator Array length operator

Semantic: #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH \n
DB name: \c "mdl::operator_len(<0>[])" \n
MDL name: \c "operator_len(<0>[])"

The following requirements apply when creating function calls of the array length operator:
- the expression list for the arguments contains a single expression named \c "a" of type
  #mi::neuraylib::IType_array.


\subsection mi_neuray_mdl_array_index_operator Array index operator

Semantic: #mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX \n
DB name: \c "mdl::operator[](<0>[],int)" \n
MDL name: \c "operator[](<0>[],int)"

Despite its name, the array index operator can also be used on vectors and matrices.

The following requirements apply when creating function calls of the array index operator:
- the expression list for the arguments contains two expressions named \c "a" and \c "i",
  respectively,
- the expression named \c "a" is of type #mi::neuraylib::IType_array, #mi::neuraylib::IType_vector,
  or #mi::neuraylib::IType_matrix, and
- the expression named \c "i" is of type #mi::neuraylib::IType_int.


\subsection mi_neuray_mdl_ternary_operator Ternary operator

Semantic: #mi::neuraylib::IFunction_definition::DS_TERNARY \n
DB name: \c "mdl::operator?(bool,<0>,<0>)" \n
MDL name: \c "operator?(bool,<0>,<0>)"

The following requirements apply when creating function calls of the ternary operator:
- the expression list for the arguments contains three expressions named \c "cond", \c "true_exp",
   and \c "false_exp", respectively,
- the expression named \c "cond" is of type #mi::neuraylib::IType_bool, and
- the the two other expressions are of the same type.


\subsection mi_neuray_mdl_cast_operator Cast operator

Semantic: #mi::neuraylib::IFunction_definition::DS_CAST \n
DB name: \c "mdl::operator_cast(<0>)" \n
MDL name: \c "operator_cast(<0>)"

The following requirements apply when creating function calls of the ternary operator:
- the expression list for the arguments contains \em two expressions named \c "cast" and
  \c "cast_return", respectively,
- the expression named \c "cast" is of an arbitrary type (this is the intended argument of the
  function call),
- the type of the expression named \c "cast_return" specifies the return type of the function
  call (the expression itself is not used), and
- the types of \c "cast" and \c "cast_return" are compatible.

See also #mi::neuraylib::IExpression_factory::create_cast().

@}*/ // end group mi_neuray_mdl_elements

/** \addtogroup mi_neuray_mdl_elements
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
    MDL_VERSION_INVALID = 0xffffffffU,     ///< Invalid MDL version
    MDL_VERSION_FORCE_32_BIT = 0xffffffffU // Undocumented, for alignment only
};

/// This interface represents an MDL module.
///
/// \see #mi::neuraylib::IMaterial_definition, #mi::neuraylib::IMaterial_instance
/// \see #mi::neuraylib::IFunction_definition, #mi::neuraylib::IFunction_call
class IModule : public
    mi::base::Interface_declare<0xe283b0ee,0x712b,0x4bdb,0xa2,0x13,0x32,0x77,0x7a,0x98,0xf9,0xa6,
                                neuraylib::IScene_element>
{
public:
    /// Returns the name of the MDL source file from which this module was created.
    ///
    /// \return         The full pathname of the source file from which this MDL module was created,
    ///                 or \c NULL if no such file exists.
    virtual const char* get_filename() const = 0;

    /// Returns the MDL name of the module.
    ///
    /// \note The MDL name of the module is different from the name of the DB element.
    ///       Use #mi::neuraylib::ITransaction::name_of() to obtain the name of the DB element.
    ///
    /// \return         The MDL name of the module.
    virtual const char* get_mdl_name() const = 0;

    /// Returns the number of package components in the MDL name.
    virtual Size get_mdl_package_component_count() const = 0;

    /// Returns the name of a package component in the MDL name.
    ///
    /// \return         The \p index -th package component name, or \c NULL if \p index is out of
    ///                 bounds.
    virtual const char* get_mdl_package_component_name( Size index) const = 0;

    /// Returns the simple MDL name of the module.
    ///
    /// The simple name is the last component of the MDL name, i.e., without any packages and scope
    /// qualifiers.
    ///
    /// \return         The simple MDL name of the module.
    virtual const char* get_mdl_simple_name() const = 0;

    /// Returns the MDL version of this module.
    virtual Mdl_version get_mdl_version() const = 0;

    /// Returns the number of modules imported by the module.
    virtual Size get_import_count() const = 0;

    /// Returns the DB name of the imported module at \p index.
    ///
    /// \param index    The index of the imported module.
    /// \return         The DB name of the imported module.
    virtual const char* get_import( Size index) const = 0;

    /// Returns the types exported by this module.
    virtual const IType_list* get_types() const = 0;

    /// Returns the constants exported by this module.
    virtual const IValue_list* get_constants() const = 0;

    /// Returns the number of function definitions exported by the module.
    virtual Size get_function_count() const = 0;

    /// Returns the DB name of the function definition at \p index.
    ///
    /// \param index    The index of the function definition.
    /// \return         The DB name of the function definition. The method may return \c NULL for
    ///                 valid indices if the corresponding function definition has already been
    ///                 removed from the DB.
    virtual const char* get_function( Size index) const = 0;

    /// Returns the number of material definitions exported by the module.
    virtual Size get_material_count() const = 0;

    /// Returns the DB name of the material definition at \p index.
    ///
    /// \param index    The index of the material definition.
    /// \return         The DB name of the material definition. The method may return \c NULL for
    ///                 valid indices if the corresponding material definition has already been
    ///                 removed from the DB.
    virtual const char* get_material( Size index) const = 0;

    /// Returns the number of resources defined in the module.
    /// Resources defined in a module that is imported by this module are not included.
    virtual Size get_resources_count() const = 0;

    /// Returns the type of the resource at \p index.
    ///
    /// \param index    The index of the resource.
    /// \return         The type of the resource.
    virtual const IType_resource* get_resource_type( Size index) const = 0;

    /// Returns the absolute MDL file path of the resource at \p index.
    ///
    /// \param index    The index of the resource.
    /// \return         The absolute MDL file path of the resource.
    virtual const char* get_resource_mdl_file_path( Size index) const = 0;

    /// Returns the database name of the resource at \p index.
    ///
    /// \param index    The index of the resource.
    /// \return         The database name of the resource or \c NULL if
    ///                 this resource could not be resolved.
    virtual const char* get_resource_name( Size index) const = 0;

    /// Returns the number of annotations defined in the module.
    virtual Size get_annotation_definition_count() const = 0;

    /// Returns the annotation definition at \p index.
    ///
    /// \param index    The index of the annotation definition.
    /// \return         The annotation definition or \c NULL if
    ///                 \p index is out of range.
    virtual const IAnnotation_definition* get_annotation_definition( Size index) const = 0;

    /// Returns the annotation definition of the given \p name.
    ///
    /// \param name     The name of the annotation definition.
    /// \return         The annotation definition or \c NULL if there is no such definition.
    virtual const IAnnotation_definition* get_annotation_definition( const char* name) const = 0;

    /// Returns the annotations of the module, or \c NULL if there are no such annotations.
    virtual const IAnnotation_block* get_annotations() const = 0;

    /// Indicates whether this module is a standard module.
    ///
    /// Examples for standard modules are \c "limits", \c "anno", \c "state", \c "math", \c "tex",
    /// \c "noise", and \c "df".
    virtual bool is_standard_module() const = 0;

    /// Indicates whether this module results from an \c .mdle file.
    virtual bool is_mdle_module() const = 0;

    /// Returns overloads of a function definition.
    ///
    /// The method returns overloads of function definition of this module, either all overloads or
    /// just the overloads matching a given set of arguments.
    ///
    /// \param name             The DB name of a function definition from this module \em without
    ///                         signature.
    /// \param arguments        Optional arguments to select specific overload(s). If present, the
    ///                         method returns only the overloads of \p name whose signature matches
    ///                         the provided arguments, i.e., a call to
    ///                         #mi::neuraylib::IFunction_definition::create_function_call() with
    ///                         these arguments would succeed.
    /// \return                 The DB names of overloads of the given function definition, or
    ///                         \c NULL if \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const IExpression_list* arguments = 0) const = 0;

    /// Returns overloads of a function definition.
    ///
    /// The method returns the best-matching overloads of a function definition of this module,
    /// given a list of positional parameter types.
    ///
    /// \note This overload should only be used if no actual arguments are available. If arguments
    ///       are available, consider using
    ///       #get_function_overloads(const char*,const IExpression_list*)const instead.
    ///
    /// \note This method does not work for the function definitions with the following semantics:
    ///       - #mi::neuraylib::IFunction_definition::DS_CAST,
    ///       - #mi::neuraylib::IFunction_definition::DS_TERNARY,
    ///       - #mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX,
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH. and
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS.
    ///       These are the \ref mi_neuray_mdl_template_like_functions_definitions plus the field
    ///       access function definitions.
    ///
    /// \param name             The DB name of a function definition from this module \em without
    ///                         signature.
    /// \param parameter_types  A static or dynamic array with elements of type #mi::IString
    ///                         representing positional parameter type names as returned by
    ///                         #mi::neuraylib::IFunction_definition::get_mdl_parameter_type_name().
    /// \return                 The DB names of overloads of the given function definition, or
    ///                         \c NULL if \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const IArray* parameter_types) const = 0;

    /// Returns true if all imports of the module are valid.
    ///
    /// \param context     In case of failure, the execution context can be checked for error
    ///                    messages. Can be \c NULL.
    virtual bool is_valid( IMdl_execution_context* context) const = 0;

    /// Reload the module from disk.
    ///
    /// \note This function works for file-based modules, only.
    ///
    /// \param context     In case of failure, the execution context can be checked for error
    ///                    messages. Can be \c NULL.
    /// \param recursive   If true, all imported file based modules are reloaded
    ///                    prior to this one.
    /// \return
    ///               -     0: Success
    ///               -    -1: Reloading failed, check the context for details.
    virtual Sint32 reload( bool recursive, IMdl_execution_context* context) = 0;

    /// Reload the module from string.
    ///
    /// \note This function works for string/memory-based modules, only. Standard modules and
    /// the built-in \if MDL_SOURCE_RELEASE module mdl::base \else modules \c mdl::base and
    /// \c mdl::nvidia::distilling_support \endif cannot be reloaded.
    ///
    /// \param module_source The module source code.
    /// \param recursive     If true, all imported file based modules are reloaded
    ///                      prior to this one.
    /// \param context       In case of failure, the execution context can be checked for error
    ///                      messages. Can be \c NULL.
    /// \return
    ///               -     0: Success
    ///               -    -1: Reloading failed, check the context for details.
    virtual Sint32 reload_from_string(
        const char* module_source,
        bool recursive,
        IMdl_execution_context* context) = 0;

    virtual const IArray* deprecated_get_function_overloads(
        const char* name, const char* param_sig) const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_11_1
    inline const IArray* get_function_overloads(
        const char* name, const char* param_sig) const
    {
        return deprecated_get_function_overloads( name, param_sig);
    }
#endif
};

/*@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMODULE_H

