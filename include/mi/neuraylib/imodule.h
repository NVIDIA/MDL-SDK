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
/// \brief      Scene element Module

#ifndef MI_NEURAYLIB_IMODULE_H
#define MI_NEURAYLIB_IMODULE_H

#include <mi/neuraylib/iexpression.h>
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
DB elements of type #mi::neuraylib::IFunction_definition are created in the DB accordingly. Both,
material and function definitions can be instantiated. Those instantiations are represented by the
interface #mi::neuraylib::IFunction_call.



\section mi_mdl_names Naming scheme for MDL elements

There are four different types of names for MDL elements: DB names, MDL names, simple MDL names,
and serialized names.


\subsection mi_mdl_encoded_names Encoded names

Encoded names are a naming scheme for MDL elements. Its purpose is to avoid certain problems
that exist with the old naming scheme, especially in the context of package and module names
containing certain meta-characters. For example, given the function name
\c "::f::g(int)::h::i(j::k)": does the signature start at the first or second left parenthesis?

The main idea is to use percent-encoding for certain characters when they do \em not appear as
meta-character.

<table>
  <tr><th>Decoded character</th><th>Encoded character</th></tr>
  <tr><td><code>(</code></td>   <td><code>%28</code></td></tr>
  <tr><td><code>)</code></td>   <td><code>%29</code></td></tr>
  <tr><td><code>&lt;</code></td><td><code>%3C</code></td></tr>
  <tr><td><code>&gt;</code></td><td><code>%3E</code></td></tr>
  <tr><td><code>,</code></td>   <td><code>%2C</code></td></tr>
  <tr><td><code>:</code></td>   <td><code>%3A</code></td></tr>
  <tr><td><code>$</code></td>   <td><code>%24</code></td></tr>
  <tr><td><code>#</code></td>   <td><code>%23</code></td></tr>
  <tr><td><code>?</code></td>   <td><code>%3F</code></td></tr>
  <tr><td><code>\@</code></td>  <td><code>%40</code></td></tr>
  <tr><td><code>%</code></td>   <td><code>%25</code></td></tr>
</table>

To avoid redundant representations, only upper-case letters are used as hexadecimal digits of
encoded characters. All characters not listed in the table above are never encoded.

Encoding these characters when they do \em not appear as meta-characters avoids the known
ambiguities. The example from above would be either
\c "::f::g(int%29::h::i%28j::k)" (if the signature starts at the first left parenthesis) or
\c "::f::g%28int%29::h::i(j::k)" (if the signature starts at the second left parenthesis).

An important consequence from resolving such ambiguities is the fact that it is always possible to
decode an encoded name, whereas (in general) it is \em not possible to encode a name without
context. Therefore, you should use encoded names as much as possible, and use decoded names \em
only e.g. for display purposes.

\note All DB names of modules, material and function definitions, their MDL names and simple names,
and all MDL-related type names in the API are encoded; exceptions from this rule are explicitly
documented. One such exception are the functions
#mi::neuraylib::IMdl_factory::decode_name(),
#mi::neuraylib::IMdl_factory::encode_module_name(),
#mi::neuraylib::IMdl_factory::encode_function_definition_name(), and
#mi::neuraylib::IMdl_factory::encode_type_name(), which help with decoding and encoding of names.

\note Due to a technical limitation module names containing parentheses or commas are not yet
supported.


\subsection mi_mdl_db_names DB names

DB names are the most prominent type of names for MDL elements in the \neurayApiName. These type
names are mainly used to identify MDL elements, in particular, to access them in the database.

The DB names for modules as well as for function and material definitions have an \c "mdl" or \c
"mdle" prefix. This prefix is also used for automatically created function calls and materials
instances used as parameter defaults. User-generated function calls, material instances, and
compiled materials might use the prefix, but are not required to.

If the interface of an MDL element is given, then its DB name can be obtained from
#mi::neuraylib::ITransaction::name_of().

Decoded DB names are shown in the table below only for better understanding and comparison with
subsequent tables. There is hardly any use of them, except for dealing with older file formats.

<table>
  <tr>
    <th>MDL entity</th>
    <th>Decoded name</th>
    <th>Encoded name</th>
  </tr>
  <tr>
    <td>module with builtins</td>
    <td><code>mdl::&lt;builtins&gt;</code></td>
    <td><code>mdl::%3Cbuiltins%3E</code></td>
  </tr>
  <tr>
    <td>user module</td>
    <td><code>mdl::foo</code></td>
    <td><code>mdl::foo</code></td>
  </tr>
  <tr>
    <td>MDLE module (Linux)</td>
    <td><code>mdle::/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(1)</td>
    <td><code>mdle::/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(1)</td>
  </tr>
  <tr>
    <td>MDLE module (Windows)</td>
    <td><code>mdle::/C:/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(1)</td>
    <td><code>mdle::/C%3A/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(1)</td>
  </tr>
  <tr>
    <td>builtin function</td>
    <td><code>mdl::operator+(float,float)</code></td>
    <td><code>mdl::operator+(float,float)</code></td>
  </tr>
  <tr>
    <td>builtin function, affected by encoding</td>
    <td><code>mdl::operator<<(int,int)</code></td>
    <td><code>mdl::operator%3C%3C(int,int)</code></td>
  </tr>
  <tr>
    <td>builtin template-like function</td>
    <td><code>mdl::operator?(bool,<0>,<0>)</code></td>
    <td><code>mdl::operator%3F(bool,%3C0%3E,%3C0%3E)</code></td>
  </tr>
  <tr>
    <td>user function</td>
    <td><code>mdl::foo::my_func(color)</code></td>
    <td><code>mdl::foo::my_func(color)</code></td>
  </tr>
  <tr>
    <td>user function, module name affected by encoding</td>
    <td><code>mdl::foo,bar::my_func(::foo,bar::my_enum)</code></td>
    <td><code>mdl::foo%2Cbar::%my_func(::%foo%2Cbar::%my_enum)</code></td>
  </tr>
  <tr>
    <td>user material</td>
    <td><code>mdl::foo::my_mat(color)</code></td>
    <td><code>mdl::foo::my_mat(color)</code></td>
  </tr>
</table>

(1) Note that the DB name for an MDLE module is \em not the same as the filename (modulo
    \c "mdle::" prefix), in particular on Windows (slash before drive letter, encoded colon after
    drive letter, slashes vs backslashes). Even on non-Windows systems there might be differences
    due to filename normalization. Use #mi::neuraylib::IMdl_factory::get_db_module_name() to obtain
    the DB name for an MDLE module from the filename.


\subsection mi_mdl_mdl_names MDL names

MDL names are useful for display purposes. They are similar to the DB names, except that they lack
the \c "mdl" or \c "mdle" prefix. Entities from the \c ::&lt;builtins&gt; module also lack the
leading scope \c "::".

The interfaces for modules as well as for function and material definitions provide methods to
obtain the MDL name, see #mi::neuraylib::IModule::get_mdl_name(), and
#mi::neuraylib::IFunction_definition::get_mdl_name(). The corresponding DB name can be obtained
from an MDL name with the help of the methods #mi::neuraylib::IMdl_factory::get_db_module_name() and
#mi::neuraylib::IMdl_factory::get_db_definition_name(). Note that there is no MDL name for function
calls, material instances, or compiled materials.

For display purposes you might want to decode the MDL name using
#mi::neuraylib::IMdl_factory::decode_name().

<table>
  <tr>
    <th>MDL entity</th>
    <th>Decoded name</th>
    <th>Encoded name</th>
  </tr>
  <tr>
    <td>module with builtins</td>
    <td><code>::&lt;builtins&gt;</code></td>
    <td><code>::%3Cbuiltins%3E</code></td>
  </tr>
  <tr>
    <td>user module</td>
    <td><code>::%foo</code></td>
    <td><code>::%foo</code></td>
  </tr>
  <tr>
    <td>MDLE module (Linux)</td>
    <td><code>::/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(2)</td>
    <td><code>::/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(2)</td>
  </tr>
  <tr>
    <td>MDLE module (Windows)</td>
    <td><code>::/C:/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(2)</td>
    <td><code>::/C%3A/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(2)</td>
  </tr>
  <tr>
    <td>builtin function</td>
    <td><code>operator+(float,float)</code></td>
    <td><code>operator+(float,float)</code></td>
  </tr>
  <tr>
    <td>builtin function, affected by encoding</td>
    <td><code>operator<<(int,int)</code></td>
    <td><code>operator%3C%3C(int,int)</code></td>
  </tr>
  <tr>
    <td>builtin template-like function</td>
    <td><code>operator?(bool,<0>,<0>)</code></td>
    <td><code>operator%3F(bool,%3C0%3E,%3C0%3E)</code></td>
  </tr>
  <tr>
    <td>user function</td>
    <td><code>::%foo::my_func(color)</code></td>
    <td><code>::%foo::my_func(color)</code></td>
  </tr>
  <tr>
    <td>user function, module name affected by encoding</td>
    <td><code>::%foo,bar::my_func(::%foo,bar::my_enum)</code></td>
    <td><code>::%foo%2Cbar::%my_func(::%foo%2Cbar::%my_enum)</code></td>
  </tr>
  <tr>
    <td>user material</td>
    <td><code>::%foo::my_mat(color)</code></td>
    <td><code>::%foo::my_mat(color)</code></td>
  </tr>
</table>

(2) Note that the MDL name for an MDLE module is \em not the same as the filename (modulo \c "::"
    prefix), in particular on Windows (slash before drive letter, encoded colon after drive
    letter, slashes vs backslashes). Even on non-Windows systems there might be differences due to
    filename normalization.


\subsection mi_mdl_simple_mdl_names Simple MDL names

Simple MDL names are a variant of the MDL names above. They are used in a few places when the
context is clear. Simple MDL names for modules lack the package name prefix including the scope
separator. Simple MDL names for function, material, and annotation definitions lack the module name
prefix including the scope separator and the signature suffix.

Note that due to function overloading, multiple functions within a module might share the same
simple name.

The interfaces for modules as well as for function and material definitions provide methods to
obtain the simple MDL name, see #mi::neuraylib::IModule::get_mdl_simple_name(), and
#mi::neuraylib::IFunction_definition::get_mdl_simple_name().

For display purposes you might want to decode the simple MDL name using
#mi::neuraylib::IMdl_factory::decode_name().

<table>
  <tr>
    <th>MDL entity</th>
    <th>Decoded name</th>
    <th>Encoded name</th>
  </tr>
  <tr>
    <td>module with builtins</td>
    <td><code>&lt;builtins&gt;</code></td>
    <td><code>%3Cbuiltins%3E</code></td>
  </tr>
  <tr>
    <td>user module</td>
    <td><code>foo</code></td>
    <td><code>foo</code></td>
  </tr>
  <tr>
    <td>MDLE module (Linux)</td>
    <td><code>/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(3)</td>
    <td><code>/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(3)</td>
  </tr>
  <tr>
    <td>MDLE module (Windows)</td>
    <td><code>/C:/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(3)</td>
    <td><code>/C%3A/path/to/bar.mdle</code>&nbsp;&nbsp;&nbsp;(3)</td>
  </tr>
  <tr>
    <td>builtin function</td>
    <td><code>operator+</code></td>
    <td><code>operator+</code></td>
  </tr>
  <tr>
    <td>builtin function, affected by encoding</td>
    <td><code>operator<<</code></td>
    <td><code>operator%3C%3C</code></td>
  </tr>
  <tr>
    <td>builtin template-like function</td>
    <td><code>operator?</code></td>
    <td><code>operator%3F</code></td>
  </tr>
  <tr>
    <td>user function</td>
    <td><code>my_func</code></td>
    <td><code>my_func</code></td>
  </tr>
  <tr>
    <td>user function, module name affected by encoding</td>
    <td><code>my_func</code></td>
    <td><code>my_func</code></td>
  </tr>
  <tr>
    <td>user material</td>
    <td><code>my_mat</code></td>
    <td><code>my_mat</code></td>
  </tr>
</table>

(3) Note that the simple MDL name for an MDLE module is \em not the same as the filename, in
    particular on Windows (slash before drive letter, encoded colon after drive letter, slashes vs
    backslashes). Even on non-Windows systems there might be differences due to filename
    normalization.


\subsection mi_mdl_serialized_names Serialized names

Serialized names are identical to encoded DB names with the exception of \ref
mi_neuray_mdl_template_like_function_definitions. For these functions, the serialized names have a
suffix in angle brackets that contains additional information about the template parameters. This
extra information is useful to reconstruct the correct template instance upon deserialization.
See \ref mi_neuray_mdl_template_like_function_definitions for an example of serialized names for
each template-like function.

The methods #mi::neuraylib::IMdl_impexp_api::serialize_function_name(),
#mi::neuraylib::IMdl_impexp_api::deserialize_function_name(), and
#mi::neuraylib::IMdl_impexp_api::deserialize_module_name() deal with serialized names.

Serialized names occur only in the encoded form.

<table>
  <tr>
    <th>MDL entity</th>
    <th>Decoded name</th>
    <th>Encoded name</th>
  </tr>
  <tr>
    <td>module with builtins</td>
    <td>&mdash;</td>
    <td><code>mdl::%3Cbuiltins%3E</code></td>
  </tr>
  <tr>
    <td>user module</td>
    <td>&mdash;</td>
    <td><code>mdl::foo</code></td>
  </tr>
  <tr>
    <td>MDLE module (Linux)</td>
    <td>&mdash;</td>
    <td><code>mdle::bar.mdle</code>&nbsp;&nbsp;&nbsp;(4)</td>
  </tr>
  <tr>
    <td>MDLE module (Windows)</td>
    <td>&mdash;</td>
    <td><code>mdle::bar.mdle</code>&nbsp;&nbsp;&nbsp;(5)</td>
  </tr>
  <tr>
    <td>builtin function</td>
    <td>&mdash;</td>
    <td><code>mdl::operator+(float,float)</code></td>
  </tr>
  <tr>
    <td>builtin function, affected by encoding</td>
    <td>&mdash;</td>
    <td><code>mdl::operator%3C%3C(int,int)</code></td>
  </tr>
  <tr>
    <td>builtin template-like function</td>
    <td>&mdash;</td>
    <td><code>mdl::operator%3F(bool,%3C0%3E,%3C0%3E)&lt;int&gt;</code></td>
  </tr>
  <tr>
    <td>user function</td>
    <td>&mdash;</td>
    <td><code>mdl::foo::my_func(color)</code></td>
  </tr>
  <tr>
    <td>user function, module name affected by encoding</td>
    <td>&mdash;</td>
    <td><code>mdl::%foo%2Cbar::%my_func(::%foo%2Cbar::%my_enum)</code></td>
  </tr>
  <tr>
    <td>user material</td>
    <td>&mdash;</td>
    <td><code>mdl::foo::my_mat(color)</code></td>
  </tr>
</table>

(4) With an MDLE callback that strips all directory components during serialization (see
    #mi::neuraylib::IMdle_serialization_callback). \n
(5) With an MDLE callback that strips all drive and directory components during serialization (see
    #mi::neuraylib::IMdle_serialization_callback).



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
an intervening dot, e.g., \c "foo.param_int". The function definition has a single parameter \c "s"
of the struct type and the corresponding field type as return type.

\par Example
The MDL code
\code
export struct Foo {
    int param_int;
    float param_float = 0;
};
\endcode
in a module \c "mod_struct" creates the two function definitions named \c
"mdl::mod_struct::Foo.param_int(::mod_struct::Foo)" and \c
"mdl::mod_struct::Foo.param_float(::mod_struct::Foo)" to represent the member selection operators
\c "Foo.param_int" and \c "Foo.param_float". The function definitions have a single parameter \c
"s" of type \c "mdl::mod_struct::Foo" and return type
#mi::neuraylib::IType_int and #mi::neuraylib::IType_float, respectively.

\section mi_neuray_mdl_arrays Arrays

In contrast to struct types which are explicitly declared there are infinitely many array types
(considering pairs of element type and array length). Deferred-sized arrays make the situation even
more complicated. Each of these array types would have its own constructor and index operator.
Therefore, template-like functions definitions are used in the context of arrays to satisfy this
need. See the next section for details

\section mi_neuray_mdl_template_like_function_definitions Template-like function definitions

Usually, function definitions have a fixed set of parameter types and a fixed return type.
Exceptions of this rule are the following six function definitions which rather have the character
of template functions with generic parameter and/or return types.

- the array constructor (#mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR),
- the array length operator (#mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH),
- the array index operator (#mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX),
- the ternary operator (#mi::neuraylib::IFunction_definition::DS_TERNARY), and
- the cast operator (#mi::neuraylib::IFunction_definition::DS_CAST).
- the decl_cast operator (#mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_DECL_CAST).

The MDL and DB names of these function definitions use \c "<0>" or \c "T" to indicate such a generic
parameter or return type. When querying the actual type, #mi::neuraylib::IType_int (arbitrary
choice) is returned for lack of a better alternative.

When creating function calls from such template-like function definitions, the parameter and return
types are fixed, i.e., the function call itself has concrete parameter and return types as usual,
and has no template-like behavior as the definition from which it was created. This implies that,
for example, after creating a ternary operator on floats, you cannot set its arguments to
expressions of a different type than float (this would require creation of another function call
with the desired parameter types).

Template-like functions are those functions for which serialized name and DB name differ (see
\ref mi_mdl_names).

More details about the six different template-like function definitions follow.


\subsection mi_neuray_mdl_array_constructor Array constructor

Semantic: #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR \n
DB name: \c "mdl::T[](...)" \n
MDL name: \c "T[](...)" \n
Serialized named (example): \c "mdl::T[](...)<int,42>"

The following requirements apply when creating function calls of the array constructor:
- the expression list for the arguments contains a non-zero number of arguments
  named "value0", "value1", and so on,
- all arguments must be of the same type (which is the element type of the constructed array).

The suffix for the serialized name has two arguments, the type name of the element type, and the
size of the array.

\subsection mi_neuray_mdl_array_length_operator Array length operator

Semantic: #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH \n
DB name: \c "mdl::operator_len(%3C0%3E[])" \n
MDL name): \c "operator_len(%3C0%3E[])" \n
Serialized name (example): \c "operator_len(%3C0%3E[])<float[42]>"

The following requirements apply when creating function calls of the array length operator:
- the expression list for the arguments contains a single expression named \c "a" of type
  #mi::neuraylib::IType_array.

The suffix for the serialized name has one argument, the type name of the array.


\subsection mi_neuray_mdl_array_index_operator Array index operator

Semantic: #mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX \n
DB name: \c "mdl::operator[](%3C0%3E[],int)" \n
MDL name: \c "operator[](%3C0%3E[],int)" \n
Serialized name (example): \c "operator[](%3C0%3E[],int)<float[42]>"

Despite its name, the array index operator can also be used on vectors and matrices.

The following requirements apply when creating function calls of the array index operator:
- the expression list for the arguments contains two expressions named \c "a" and \c "i",
  respectively,
- the expression named \c "a" is of type #mi::neuraylib::IType_array, #mi::neuraylib::IType_vector,
  or #mi::neuraylib::IType_matrix, and
- the expression named \c "i" is of type #mi::neuraylib::IType_int.

The suffix for the serialized name has one argument, the type name of the array, vector, or matrix.


\subsection mi_neuray_mdl_ternary_operator Ternary operator

Semantic: #mi::neuraylib::IFunction_definition::DS_TERNARY \n
DB name: \c "mdl::operator%3F(bool,%3C0%3E,%3C0%3E)" \n
MDL name: \c "operator%3F(bool,%3C0%3E,%3C0%3E)" \n
Serialized name (example): \c "operator%3F(bool,%3C0%3E,%3C0%3E)<float>"

The following requirements apply when creating function calls of the ternary operator:
- the expression list for the arguments contains three expressions named \c "cond", \c "true_exp",
   and \c "false_exp", respectively,
- the expression named \c "cond" is of type #mi::neuraylib::IType_bool, and
- the two other expressions are of the same type.

The suffix for the serialized name has one argument, the type name of the \c "true_exp" expression
(which is equal to the type name of the \c "false_exp" expression).


\subsection mi_neuray_mdl_cast_operator Cast operator

Semantic: #mi::neuraylib::IFunction_definition::DS_CAST \n
DB name: \c "mdl::operator_cast(%3C0%3E)" \n
MDL name: \c "operator_cast(%3C0%3E)" \n
Serialized name (example):
  \c "operator_cast(%3C0%3E)<::foo::my_enum,::bar::my_enum>"

The following requirements apply when creating function calls of the cast operator:
- the expression list for the arguments contains \em two expressions named \c "cast" and
  \c "cast_return", respectively,
- the expression named \c "cast" is of an arbitrary type (this is the intended argument of the
  function call),
- the type of the expression named \c "cast_return" specifies the return type of the function
  call (no frequency qualifiers; the expression itself is not used), and
- the types of \c "cast" and \c "cast_return" are compatible.

The suffix for the serialized name has two arguments, the type name of the \c "cast" expression,
followed by the type name of the \c "cast_return" expression.

See also #mi::neuraylib::IExpression_factory::create_cast().


\subsection mi_neuray_mdl_decl_cast_operator Decl_cast operator

The \c decl_cast operator is used to express casts between different struct types
from the same struct category.

Semantic: #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_DECL_CAST \n
DB name: \c "mdl::operator_decl_cast(%3C0%3E)" \n
MDL name: \c "operator_decl_cast(%3C0%3E)" \n
Serialized name (example):
  \c "operator_decl_cast(%3C0%3E)<::foo::some_material_type,::foo::another_material_type>"

The following requirements apply when creating function calls of the decl_cast operator:
- the expression list for the arguments contains \em two expressions named \c "cast" and
  \c "cast_return", respectively,
- the expression named \c "cast" is a struct type with a struct category (this is
  the intended argument of the function call),
- the type of the expression named \c "cast_return" specifies the return type of the function
  call (no frequency qualifiers; the expression itself is not used), and
- the types of \c "cast" and \c "cast_return" are struct types from the same struct category.

The suffix for the serialized name has two arguments, the type name of the \c "cast" expression,
followed by the type name of the \c "cast_return" expression.


@}*/ // end group mi_neuray_mdl_elements

/** \addtogroup mi_neuray_mdl_elements
@{
*/

/// This interface represents an MDL module.
///
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

    /// Returns the struct categories exported by this module.
    virtual const IStruct_category_list* get_struct_categories() const = 0;

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
    ///
    /// Resources defined in a module that is imported by this module are not included.
    virtual Size get_resources_count() const = 0;

    /// Returns a resource defined in the module.
    ///
    /// Resources defined in a module that is imported by this module are not included.
    virtual const IValue_resource* get_resource( Size index) const = 0;

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

    /// Returns overloads of a function or material definition.
    ///
    /// The method returns overloads of a function or material definition of this module, either
    /// all overloads or just the overloads matching a given set of arguments.
    ///
    /// \param name             The simple name or the DB name \em without signature of a function
    ///                         or material definition from this module.
    /// \param arguments        Optional arguments to select specific overload(s). If present, the
    ///                         method returns only the overloads of \p name whose signature
    ///                         matches the provided arguments, i.e., a call to
    ///                         #mi::neuraylib::IFunction_definition::create_function_call() with
    ///                         these arguments would succeed.
    /// \return                 The DB names of overloads of the given function or material
    ///                         definition, or \c NULL if \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const IExpression_list* arguments = 0) const = 0;

    /// Returns overloads of a function or material definition.
    ///
    /// The method returns the best-matching overloads of a function or material definition of this
    /// module, given a list of positional parameter types.
    ///
    /// \note This overload should only be used if no actual arguments are available. If arguments
    ///       are available, consider using
    ///       #get_function_overloads(const char*,const IExpression_list*)const instead.
    ///
    /// \note This method does not work for the function definitions with the following semantics:
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH,
    ///       - #mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX,
    ///       - #mi::neuraylib::IFunction_definition::DS_TERNARY,
    ///       - #mi::neuraylib::IFunction_definition::DS_CAST,
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_DECL_CAST, and
    ///       - #mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIELD_ACCESS.
    ///       These are the \ref mi_neuray_mdl_template_like_function_definitions plus the field
    ///       access function definitions.
    ///
    /// \param name             The simple name or the DB name \em without signature of a function
    ///                         or material definition from this module.
    /// \param parameter_types  A static or dynamic array with elements of type #mi::IString
    ///                         representing positional parameter type names as returned by
    ///                         #mi::neuraylib::IFunction_definition::get_mdl_parameter_type_name().
    /// \return                 The DB names of overloads of the given function or material
    ///                         definition, or \c NULL if \p name is invalid.
    virtual const IArray* get_function_overloads(
        const char* name, const IArray* parameter_types) const = 0;

    /// Returns \c true if all imports of the module are valid.
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
    /// \param recursive   If \c true, all imported file-based modules are reloaded
    ///                    prior to this one.
    /// \return
    ///               -     0: Success
    ///               -    -1: Reloading failed, check the context for details.
    virtual Sint32 reload( bool recursive, IMdl_execution_context* context) = 0;

    /// Reload the module from string.
    ///
    /// \note This function works for string/memory-based modules, only. Standard modules and the
    /// built-in modules \c mdl::base and \c mdl::nvidia::distilling_support cannot be reloaded.
    ///
    /// \param module_source The module source code.
    /// \param recursive     If \c true, all imported file-based modules are reloaded
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
};

/**@}*/ // end group mi_neuray_mdl_elements

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMODULE_H
