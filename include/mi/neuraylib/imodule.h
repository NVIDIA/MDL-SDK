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
/// \brief      Scene element Module

#ifndef MI_NEURAYLIB_IMODULE_H
#define MI_NEURAYLIB_IMODULE_H

#include <mi/neuraylib/iscene_element.h>


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
- a ternary operator named
  \c "mdl::mod_struct::operator?(bool,::mod_struct::Foo,::mod_struct::Foo)".
.
The elemental constructor has a parameter \c "param_int" of type #mi::neuraylib::IType_int and a
parameter \c "param_float" of type #mi::neuraylib::IType_float. All three function definitions have
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
more complicated. Each of these array types requires its own constructor. Therefore, a special
convention is used to represent array constructors. Conversely, handling of array constructors
requires special code paths compared to all other function calls.

Instead of infinitely many function definitions, there is \em one \em single function definition
created in the DB as placeholder for \em all array constructors. The DB name of this function
definition is \c "mdl::T[](...)". Since this function definition acts as placeholder for \em all
array constructors the reported return type, the number of parameters, and their types are
meaningless.

When creating a function call based on this function definition the following requirements must be
met: (1) the expression list for the arguments contains a non-zero number of arguments, (2) all
arguments must be of the same type (which is the element type of the constructed array), and (3) the
positional order of the arguments in the expression list is used,  their names are irrelevant.

Once a function call for the array constructor has been created its return type, the number of
arguments, and their type is fixed. Changing the array length or the element type requires the
creation of a new function call from the array constructor definition.

\section mi_neuray_mdl_indexable_types Indexable types

Arrays, vectors, and matrices are also called indexable types. For each such type occurring in an
exported type or in the signature of an exported function or material, an indexing function is
created in the DB. The name of this indexing function is formed by suffixing the name of the type
with an \em at symbol ('@'). The indexing function has two parameters, where the first parameter
\c "a" is the value to be indexed and the second parameter \c "i" is the index.

\par Example
The MDL code
\code
export int some_function(float[42] some_parameter) { ... }
export int some_function(float[<N>] some_parameter) { ... }
export int some_function(float3 some_parameter) { ... }
export int some_function(float3x3 some_parameter) { ... }
\endcode
in a module \c "mod_indexable" creates the indexing functions
- \c "mdl::mod_indexable::float[42]@(float[42],int)" with the first parameter \c "a" of type
  #mi::neuraylib::IType_array,
- \c "mdl::float[]@(float[],int)" with the first parameter \c "a" of type
  #mi::neuraylib::IType_array,
- \c "mdl::float3@(float3,int)" with the first parameter \c "a" of type
  #mi::neuraylib::IType_vector, and
- \c "mdl::float3x3@(float3x3,int)" with the first parameter \c "a" of type
  #mi::neuraylib::IType_matrix.
.
In all cases the second parameter \c "i" is of type #mi::neuraylib::IType_int. In the first three
cases the return type is #mi::neuraylib::IType_float. In the last case the return type is
#mi::neuraylib::IType_vector with three components of type #mi::neuraylib::IType_float. Note that
for immediate-sized arrays the indexing functions is in the module that uses that array type, for
all other indexable types the indexing function is in the module that defines the type.

@}*/ // end group mi_neuray_mdl_elements

/** \addtogroup mi_neuray_mdl_elements
@{
*/

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
    virtual const IAnnotation_definition* get_annotation_definition(Size index) const = 0;

    /// Returns the annotation definition of the given \p name.
    ///
    /// \param name     The name of the annotation definition.
    /// \return         The annotation definition or \c NULL if there is no such definition.
    virtual const IAnnotation_definition* get_annotation_definition(const char *name) const = 0;

    /// Returns the annotations of the module, or \c NULL if there are no such annotations.
    virtual const IAnnotation_block* get_annotations() const = 0;

    /// Indicates whether this module is a standard module.
    ///
    /// Examples for standard modules are \c "limits", \c "anno", \c "state", \c "math", \c "tex",
    /// \c "noise", and \c "df".
    virtual bool is_standard_module() const = 0;

    /// Returns overloads of a function definition.
    ///
    /// The method returns overloads of function definition of this module, either all overloads or
    /// just the overloads matching a given set of arguments.
    ///
    /// \param name        The DB name of a function definition from this module. Due to the nature
    ///                    of this method the function signature (starting with the left
    ///                    parenthesis) is irrelevant and thus optional.
    /// \param arguments   Optional arguments to select a specific overload. If present, the method
    ///                    returns only the overloads of \p name whose signature matches the
    ///                    provided arguments, i.e., a call to
    ///                    #mi::neuraylib::IFunction_definition::create_function_call() with
    ///                    these arguments would succeed.
    /// \return            The DB names of overloads of the given function definition, or \c NULL
    ///                    if \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const IExpression_list* arguments = 0) const = 0;

    /// Returns overloads of a function definition.
    ///
    /// The method returns the best-matching overloads of a function definition of this module,
    /// given a list of positional parameter types.
    ///
    /// \param name        The DB name of a function definition from this module \em without
    ///                    signature.
    /// \param param_sig   A parameter signature as a comma separated string of MDL type names. The
    ///                    method returns only the overloads of \p name whose signature matches the
    ///                    provided positional parameter types. In addition, it returns only the
    ///                    best-matching overloads according to the MDL rules for overload
    ///                    resolution. Optionally, the parameter signature can be enclosed in
    ///                    parentheses.
    /// \return            The DB names of overloads of the given function definition, or \c NULL if
    ///                    \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const char* param_sig) const = 0;

    /// Returns true if all imports of the module are valid.
    /// \param context     In case of failure, the execution context can be checked for error
    ///                    messages. Can be \c NULL.
    virtual bool is_valid(IMdl_execution_context* context) const = 0;

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
    virtual Sint32 reload(bool recursive, IMdl_execution_context* context) = 0;

    /// Reload the module from string.
    ///
    /// \note This function works for string/memory-based modules, only. Standard modules and
    /// the built-in \if MDL_SOURCE_RELEASE module mdl::base \else modules \c mdl::base and
    /// \c mdl::nvidia::distilling_support \endif cannot be reloaded.
    ///
    /// \param module_source The module source code.
    /// \param context       In case of failure, the execution context can be checked for error
    ///                      messages. Can be \c NULL.
    /// \param recursive     If true, all imported file based modules are reloaded
    ///                      prior to this one.
    /// \return
    ///               -     0: Success
    ///               -    -1: Reloading failed, check the context for details.
    virtual Sint32 reload_from_string(
        const char* module_source,
        bool recursive,
        IMdl_execution_context* context) = 0;
};

/*@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMODULE_H

