/***************************************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Module-internal utilities related to MDL scene elements.

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_UTILITIES_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_UTILITIES_H

#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/imdl_loading_wait_handle.h>

#include <base/data/db/i_db_tag.h>
#include <base/lib/robin_hood/robin_hood.h>
#include <mdl/compiler/compilercore/compilercore_visitor.h>

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_value.h"

namespace mi {
    namespace neuraylib { class IReader; }
    namespace base      { struct Uuid; }
    namespace mdl       { class IMDL; class IMDL_resource_reader; }
}

namespace MI {

namespace DB { class Transaction; class Tag; }

namespace MDL {

class Mdl_compiled_material;
class Mdl_call_resolver;
class Mdl_function_call;
class Mdl_module_wait_queue;
class Name_mangler;

// **********  Name parsing/splitting **************************************************************

// These functions are supposed to centralize all parsing/splitting of strings.
//
// More of these methods can be found in i_mdl_elements_utilities.h.
//
// Command to find all locations where parsing happens in this module (one long string without any
// line breaks):
//
// grep -E -n "(strcmp|strncmp|strchr|strrchr|strstr|find|replace|==|!=).*
//             ('\('|'\)'|'\['|'\]'|','|'\\$'|'\|'|'\\\\''|'/'|'\\\\\\\\'|':'|\"\(\"|\"\)\"|
//             \"\[\"|\"\]\"|\",\"|\"\\$\"|\"\|\"|\"'\"|\"/\"|\"\\\\\\\\\"|\":\"|\"::\"|\"mdl)"
//            *.h *.cpp
//

/// Functions on MDL names
//@{

/// Indicates whether \p name is s valid simple MDL package or module name.
bool is_valid_simple_package_or_module_name( const std::string& name);

/// Indicates whether the MDL module or definition name is absolute (starts with "::", or is from
/// an MDLE module).
bool is_absolute( const std::string& name);

/// Indicates whether \p name starts with "/".
bool starts_with_slash( const std::string& name);

/// Indicates whether the MDL definition name (without parameter types) or MDL simple name is
/// deprecated (contains "$").
bool is_deprecated( const std::string& name);

/// Indicates whether the MDL entity \p name is in module \p module_name (but not in a submodule).
///
/// Note that the check is done by string comparison, not by checking the actual module contents.
/// Also works if both parameters are core names.
bool is_in_module( const std::string& name, const std::string& module_name);

/// Removes all qualifiers if the MDL entity \p name is from module \p module_name (but not from a
/// submodule).
///
/// Note that the check is done by string comparison, not by checking the actual module contents.
std::string remove_qualifiers_if_from_module(
    const std::string& name, const std::string& module_name);

/// Removes the suffix of deprecated MDL names.
///
/// Unfortunately, we do not have enough context to know whether this name supports deprecation
/// markers (MDL definition names without parameter types) or not (e.g. module names). Try to strip
/// exactly those suffixed corresponding to MDL versions.
///
/// Names without '$' are returned as is.
std::string strip_deprecated_suffix( const std::string& name);

/// Adds a "::" prefix for builtin enum/struct type names.
std::string prefix_builtin_type_name( const char* name);

/// Removes the "::" prefix for builtin enum/struct type names.
///
/// \param check_string   Pass \p true if it is not known whether \p name is one of the builtin
///                       enum/struct type names. In this case the input is compared against the
///                       list of builtin enum/struct type names and the result is based on that.
///                       Pass \p false if the caller already knows that \p name is one of the
///                       builtin enum/struct type names (optimization).
std::string remove_prefix_for_builtin_type_name( const char* name, bool check_string = true);

// Returns the simple MDL module name from an MDL module name.
///
/// I.e., the function strips off all packages and "::" delimiters.
///
/// \param name  The MDL module name, e.g., "::p1::p2::mod".
/// \return      The simple MDL module name, e.g., "mod".
std::string get_mdl_simple_module_name( const std::string& name);

/// Returns the MDL package name (as vector of components) from a MDL module name.
///
/// I.e., removes the module name itself and splits the string at "::" delimiters.
///
/// \param name  The MDL module name, e.g., "::p1::p2::mod".
/// \return      The components of the MDL package name, e.g., [ "p1", "p2" ].
std::vector<std::string> get_mdl_package_component_names( const std::string& name);

///  Returns the simple MDL definition name from an MDL definition name.
///
/// \param name  The MDL definition (or annotation) name, without parameter types,
///              e.g., "::p1::p2::mod::fd".
/// \return      The simple MDL definition name, e.g., "fd".
std::string get_mdl_simple_definition_name( const std::string& name);

/// Returns the MDL module name from a MDL definition name.
///
/// \param name  The MDL definition (or annotation) name, without parameter types,
///              e.g., "::p1::p2::mod::fd".
/// \return      The MDL module name, e.g., "::p1::p2::mod".
std::string get_mdl_module_name( const std::string& name);

/// Returns the MDL field name from an MDL function definition name of a struct field getter
/// (DS_INTRINSIC_DAG_FIELD_ACCESS), either without parameter types, or the simple name.
///
/// \param       The MDL name of a function definition with semantic DS_INTRINSIC_DAG_FIELD_ACCESS,
///              e.g., "::p1::p2::mod::struct.field" or "struct.field".
/// \param       The MDL names of the field, e.g., "field".
std::string get_mdl_field_name( const std::string& name);

/// Splits a string at the next separator (dot or bracket pair).
///
/// \param s      The string to split.
/// \param head   The start of the string up to the dot, or the array index, or the entire string
///               if no separator is found.
/// \param tail   The remainder of the string, or empty if no separator is found.
void split_next_dot_or_bracket( const char* s, std::string& head, std::string& tail);

/// Converts an MDL annotation definition name into the name of the corresponding DB element.
///
/// Adds the prefix "mdla::".
std::string get_db_name_annotation_definition( const std::string& name);

//@}
/// Functions on DB names.
///
/// Used by overload resolution. Otherwise, use the methods on the DB elements themselves.
//@{

/// Indicates whether \p name starts with "mdl::" or "mdle::".
bool starts_with_mdl_or_mdle( const std::string& name);

/// Strips "mdl" or "mdle" for "mdl::" or "mdle::" prefixes. Returns name unchanged otherwise.
///
/// For "mdle::/.:/" prefixes (where "." is any character), it also strips the initial slash.
///
/// Basically, this method obtains the MDL name from the DB name. Usage should be limited to cases
/// where it is unavoidable, e.g., for overload resolution. Otherwise, use the methods on the DB
/// elements themselves.
std::string strip_mdl_or_mdle_prefix( const std::string& name);

//@}
/// Functions on MDL file paths (or filenames).
//@{

/// Indicates whether the MDL file path is absolute (starts with "/").
bool is_absolute_mdl_file_path( const std::string& name);

/// Returns true if the given file name ends with ".mdl".
bool has_mdl_suffix(const std::string& filename);

/// Removes the trailing ".mdl" suffix.
std::string strip_dot_mdl_suffix(  const std::string& filename);

/// Indicates whether the given filename/file path points to an (MDL) archive.
///
/// Note that the check is done by string comparison, not by checking for existence or content.
bool is_archive_filename( const std::string& filename);

/// Indicates whether the given filename/file path points to an MDLE.
///
/// Note that the check is done by string comparison, not by checking for existence or content.
bool is_mdle_filename( const std::string& filename);

/// Indicates whether the given filename/file path points to a \em member of a container (MDL
/// archive or MDLE).
///
/// Note that the check is done by string comparison, not by checking the actual container contents.
bool is_container_member( const char* filename);

/// Returns the container filename/file path for the given combined container/member
/// filename/file path, or the empty string in case of errors.
std::string get_container_filename( const char* filename);

/// Returns the member filename/file path for the given combined container/member
/// filename/file path, or the empty string in case of errors.
const char* get_container_membername( const char* filename);

/// Adds a slash at the front if the filename starts with a drive letter and colon (Windows only).
/// Otherwise, returns filename as is.
std::string add_slash_in_front_of_drive_letter( const std::string& filename);

/// Adds a slash at the front if the name starts with a drive letter and encoded colon (Windows
/// only). Otherwise, returns name as is.
std::string add_slash_in_front_of_encoded_drive_letter( const std::string& name);

/// Removes a leading slash if the input starts with slash, drive letter, colon (Windows only).
/// Otherwise, returns the input as is.
std::string remove_slash_in_front_of_drive_letter( const std::string& input);

//@}


// ********** Conversion from mi::mdl to mi::neuraylib *********************************************

/// Converts mi::mdl::IDefinition::Semantics to mi::neuraylib::IFunction_definition::Semantics.
///
/// Some values cannot appear here. Such values are mapped to DS_UNKNOWN after an assertion.
mi::neuraylib::IFunction_definition::Semantics mdl_semantics_to_ext_semantics(
    mi::mdl::IDefinition::Semantics semantic);

/// Converts mi::neuraylib::IFunction_definition::Semantics to mi::mdl::IDefinition::Semantics.
mi::mdl::IDefinition::Semantics ext_semantics_to_mdl_semantics(
    mi::neuraylib::IFunction_definition::Semantics);

/// Converts mi::mdl::IDefinition::Semantics to mi::neuraylib::IAnnotation_definition::Semantics.
///
/// Some values cannot appear here. Such values are mapped to DS_UNKNOWN after an assertion.
mi::neuraylib::IAnnotation_definition::Semantics mdl_semantics_to_ext_annotation_semantics(
    mi::mdl::IDefinition::Semantics semantic);

// ********** Conversion from mi::mdl to MI::MDL ***************************************************

/// A vector of mi::mdl::DAG_node pointer.
using Mdl_annotation_block = std::vector<const mi::mdl::DAG_node*>;

/// A vector of vectors of mi::mdl::DAG_node pointer.
using Mdl_annotation_block_vector = std::vector<std::vector<const mi::mdl::DAG_node*> >;

/// Converts mi::mdl::IType to MI::MDL::IType.
///
/// \param tf                   The type factory to use.
/// \param type                 The type to convert.
/// \param annotations          For enums and structs the annotations of the enum/struct itself,
///                             otherwise \c NULL.
/// \param member_annotations   For enums and structs the annotations of the values/fields,
///                             otherwise \c NULL.
const IType* mdl_type_to_int_type(
    IType_factory* tf,
    const mi::mdl::IType* type,
    const Mdl_annotation_block* annotations = nullptr,
    const Mdl_annotation_block_vector* member_annotations = nullptr);

/// Converts mi::mdl::IType to MI::MDL::IType.
///
/// Template version of the function above.
template <class T>
const T* mdl_type_to_int_type(
    IType_factory* tf,
    const mi::mdl::IType* type,
    const Mdl_annotation_block* annotations = nullptr,
    const Mdl_annotation_block_vector* member_annotations = nullptr)
{
    mi::base::Handle<const IType> ptr_type(
        mdl_type_to_int_type( tf, type, annotations, member_annotations));
    if( !ptr_type)
        return nullptr;
    return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
}

/// Converts mi::mdl::IType_enum to MI::MDL::IType_enum and checks,
/// if the type conflicts with an existing type.
///
/// \param tf                   The type factory to use.
/// \param type                 The type to convert.
/// \param annotations          The annotations of the enum.
/// \param member_annotations   The annotations of the values/fields.
bool mdl_type_enum_to_int_type_test(
    IType_factory* tf,
    const mi::mdl::IType_enum* type,
    const Mdl_annotation_block* annotations = nullptr,
    const Mdl_annotation_block_vector* member_annotations = nullptr);

/// Converts mi::mdl::IType_struct to MI::MDL::IType_struct and checks,
/// if the type conflicts with an existing type.
///
/// \param tf                   The type factory to use.
/// \param type                 The type to convert.
/// \param annotations          The annotations of the enum.
/// \param member_annotations   The annotations of the values/fields.
bool mdl_type_struct_to_int_type_test(
    IType_factory* tf,
    const mi::mdl::IType_struct* type,
    const Mdl_annotation_block* annotations = nullptr,
    const Mdl_annotation_block_vector* member_annotations = nullptr);

/// Converts mi::mdl::DAG_node to MI::MDL::IExpression and MI::MDL::IAnnotation
class Mdl_dag_converter
{
public:
    /// Constructor.
    ///
    /// \param ef                     The expression factory to use.
    /// \param transaction            The DB transaction to use.
    /// \param tagger                 The resource tagger to be used.
    /// \param code_dag               The DAG itself (optional). If present, used to add the
    ///                               signature for materials, otherwise DB accesses are used.
    /// \param immutable_callees      \c true for defaults, \c false for arguments
    /// \param create_direct_calls    \c true creates EK_DIRECT_CALLs, \c false creates EK_CALLs
    /// \param module_mdl_name        The MDL module name. Optional, used for localization only.
    /// \param prototype_tag          The prototype tag. Optional, used for localization only.
    /// \param resolve_resources      \c true, if resources are supposed to be loaded into the DB
    /// \param user_modules_seen      If non - \c NULL, visited user module tags and identifiers
    ///                               are put here.
    Mdl_dag_converter(
        IExpression_factory* ef,
        DB::Transaction* transaction,
        mi::mdl::IResource_tagger* tagger,
        const mi::mdl::IGenerated_code_dag* code_dag,
        bool immutable_callees,
        bool create_direct_calls,
        const char* module_mdl_name,
        DB::Tag prototype_tag,
        bool resolve_resources,
        std::set<Mdl_tag_ident>* user_modules_seen);

    /// Converts mi::mdl::IType to MI::MDL::IType.
    ///
    /// Similar to MDL::mdl_type_to_int_type(), except that it caches the results for enums and
    /// structs.
    ///
    /// \param type                 The type to convert.
    /// \param annotations          For enums and structs the annotations of the enum/struct itself,
    ///                             otherwise \c NULL.
    /// \param member_annotations   For enums and structs the annotations of the values/fields,
    ///                             otherwise \c NULL.
    const IType* mdl_type_to_int_type(
        const mi::mdl::IType* type,
        const Mdl_annotation_block* annotations = nullptr,
        const Mdl_annotation_block_vector* member_annotations = nullptr) const;

    /// Converts mi::mdl::IType to MI::MDL::IType.
    ///
    /// Template version of the function above.
    template <class T>
    const T* mdl_type_to_int_type(
        const mi::mdl::IType* type,
        const Mdl_annotation_block* annotations = nullptr,
        const Mdl_annotation_block_vector* member_annotations = nullptr) const
    {
        mi::base::Handle<const IType> ptr_type(
            mdl_type_to_int_type( type, annotations, member_annotations));
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    /// Converts mi::mdl::IValue to MI::MDL::IValue.
    ///
    /// \param type_int   The expected type of the return value. Used to control whether array
    ///                   arguments are converted to immediate-sized or deferred-sized arrays. If
    ///                   \c NULL, the type of \p value is used.
    /// \param value      The value to convert.
    /// \return           The converted value, or \c NULL in case of failures.
    IValue* mdl_value_to_int_value(
        const IType* type_int,
        const mi::mdl::IValue* value) const;

    /// Converts mi::mdl::DAG_node to MI::MDL::IExpression.
    ///
    /// \param node       The DAG node to convert.
    /// \param type_int   The expected type of the return value. Used to control whether array
    ///                   arguments are converted to immediate-sized or deferred-sized arrays. If
    ///                   \c NULL, the type of \p value is used.
    /// \return           The converted expression, or \c NULL in case of failures.
    IExpression* mdl_dag_node_to_int_expr(
        const mi::mdl::DAG_node* node,
        const IType* type_int) const;

    /// Converts a vector of mi::mdl::DAG_node pointers to MI::MDL::IAnnotation_block.
    IAnnotation_block* mdl_dag_node_vector_to_int_annotation_block(
        const Mdl_annotation_block& mdl_annotations,
        const char* qualified_name) const;

private:

    /// Find the tag for a given resource.
    DB::Tag find_resource_tag(
        const mi::mdl::IValue_resource* res) const;

    /// Converts mi::mdl::DAG_call to MI::MDL::IExpression.
    /// (creates IExpression_direct_call)
    IExpression* mdl_call_to_int_expr_direct(
        const mi::mdl::DAG_call* call, bool use_parameter_type) const;

    /// Converts mi::mdl::DAG_call to MI::MDL::IExpression.
    /// (creates IExpression_call)
    IExpression* mdl_call_to_int_expr_indirect(
        const mi::mdl::DAG_call* call, bool use_parameter_type) const;

    /// Converts mi::mdl::DAG_call to MI::MDL::IAnnotation.
    IAnnotation* mdl_dag_call_to_int_annotation(
        const mi::mdl::DAG_call* call, const char* qualified_name) const;

    /// Converts mi::mdl::DAG_node string and string array annotations
    /// to MI::MDL::IExpression, thereby translating strings according to the
    /// current locale.
    /// Identical to mdl_dag_node_to_int_expr() but for translation in the context of localization.
    IExpression* mdl_dag_node_to_int_expr_localized(
        const mi::mdl::DAG_node* argument,
        const mi::mdl::DAG_call* call,
        const IType* type_int,
        const char* qualified_name) const;

    mi::base::Handle<IExpression_factory> m_ef;
    mi::base::Handle<IValue_factory> m_vf;
    mi::base::Handle<IType_factory> m_tf;

    DB::Transaction* m_transaction;
    mi::mdl::IResource_tagger* m_tagger;
    const mi::mdl::IGenerated_code_dag* m_code_dag;
    bool m_immutable_callees;
    bool m_create_direct_calls;

    /// The MDL module name. Optional, used for localization only.
    const char* m_loc_module_mdl_name;

    /// The prototype of the converted definition. Optional, used for localization only.
    DB::Tag m_loc_prototype_tag;

    bool m_resolve_resources;

    mutable std::set<Mdl_tag_ident>* m_user_modules_seen;

    /// Cache used by mdl_type_to_int_type().
    mutable robin_hood::unordered_map<const mi::mdl::IType*, const MDL::IType*> m_cached_types;
};

/// Wrapper around mi::mdl::IGenerated_code_dag that dispatches between functions and materials.
class Code_dag
{
public:
    Code_dag( const mi::mdl::IGenerated_code_dag* code_dag, bool is_material)
      : m_code_dag( code_dag), m_is_material( is_material) { }

    /// Names (cloned and original might be nullptr)
    const char* get_name( mi::Size index);
    const char* get_simple_name( mi::Size index);
    const char* get_cloned_name( mi::Size index);
    const char* get_original_name( mi::Size index);

    /// Properties
    mi::mdl::IDefinition::Semantics get_semantics( mi::Size index);
    bool get_exported( mi::Size index);
    bool get_uniform( mi::Size index);

    /// Return type (nullptr for materials)
    const mi::mdl::IType* get_return_type( mi::Size index);

    /// Parameters
    mi::Size get_parameter_count( mi::Size index);
    const mi::mdl::IType* get_parameter_type( mi::Size index, mi::Size parameter_index);
    const char* get_parameter_type_name( mi::Size index, mi::Size parameter_index);
    const char* get_parameter_name( mi::Size index, mi::Size parameter_index);
    mi::Size get_parameter_index( mi::Size index, const char* parameter_name);
    const mi::mdl::DAG_node* get_parameter_default( mi::Size index, mi::Size parameter_index);

    /// Parameters enabled_if()
    const mi::mdl::DAG_node* get_parameter_enable_if_condition(
        mi::Size index, mi::Size parameter_index);
    mi::Size get_parameter_enable_if_condition_users( mi::Size index, mi::Size parameter_index);
    mi::Size get_parameter_enable_if_condition_user(
        mi::Size index, mi::Size parameter_index, mi::Size user_index);

    /// Parameter annotations
    mi::Size get_parameter_annotation_count( mi::Size index, mi::Size parameter_index);
    const mi::mdl::DAG_node* get_parameter_annotation(
        mi::Size index, mi::Size parameter_index, mi::Size annotation_index);

    /// Temporaries
    mi::Size get_temporary_count( mi::Size index);
    const mi::mdl::DAG_node* get_temporary( mi::Size index, mi::Size temporary_index);
    const char* get_temporary_name( mi::Size index, mi::Size temporary_index);

    /// Annotations
    mi::Size get_annotation_count( mi::Size index);
    const mi::mdl::DAG_node* get_annotation( mi::Size index, mi::Size annotation_index);
    mi::Size get_return_annotation_count( mi::Size index);
    const mi::mdl::DAG_node* get_return_annotation( mi::Size index, mi::Size annotation_index);

    /// Body
    const mi::mdl::DAG_node* get_body( mi::Size index);

    /// Hash (might be nullptr)
    const mi::mdl::DAG_hash* get_hash( mi::Size index);

private:
    const mi::mdl::IGenerated_code_dag* m_code_dag;
    bool m_is_material;
};

// ********** Conversion from MI::MDL to MI::MDL ***************************************************

/// Traverses an expression and replaces call expressions by direct call expressions.
///
/// Parameter references are resolved using \p call_context.
IExpression* int_expr_call_to_int_expr_direct_call(
    DB::Transaction* transaction,
    IExpression_factory* ef,
    const IExpression* expr,
    const std::vector<mi::base::Handle<const IExpression>>& call_context,
    Execution_context* context);

/// Traverses an expression and replaces call expressions by direct call expressions.
///
/// Parameter references are resolved using \p call_context.
IExpression* int_expr_call_to_int_expr_direct_call(
    DB::Transaction* transaction,
    IExpression_factory* ef,
    const IType* type,
    const Mdl_function_call* call,
    const std::vector<mi::base::Handle<const IExpression>>& call_context,
    Execution_context* context);

// ********** Misc utility functions around MI::MDL ************************************************

/// Indicates whether a value or an expression of a particular type can be used as an argument for a
/// parameter of a particular type.
///
/// Returns \c true iff the modifier-stripped types are identical, or if the modifier-stripped
/// argument type is an array and the modifier-stripped parameter type is a deferred-sized array of
/// the same element type.
///
/// If \p allow_compatible_types is \c true, this function also returns \c true if \p argument_type
/// can be casted to \p parameter_type. In this case, the output parameter \p needs_cast is set to
/// \cc true.
///
/// \param tf                  The type factory.
/// \param argument_type       The type of the intended argument.
/// \param parameter_type      The parameter type of the definition.
/// \param allow_cast          Indicates whether compatible types are feasible.
/// \param[out] needs_cast     Indicates whether types are compatible (but not identical), i.e.,
///                            require a cast operator (implies \c allow_cast).
bool argument_type_matches_parameter_type(
    IType_factory* tf,
    const IType* argument_type,
    const IType* parameter_type,
    bool allow_cast,
    bool &needs_cast);

/// Returns \c true iff the return type of the called function definition is varying. This includes
/// definitions where the \em effective return if varying, i.e., it is declared auto but the
/// function itself is varying.
///
/// \pre argument->get_kind() does not return IExpression::EK_TEMPORARY
///
/// Returns \c false if the referenced function call/definition has an unexpected class ID.
bool return_type_is_varying( DB::Transaction* transaction, const IExpression* argument);

/// Performs a deep copy of expressions.
///
/// "Deep copy" is defined as duplication of the DB elements referenced in calls and application of
/// the deep copy to its arguments. Note that referenced resources are not duplicated (to save
/// memory). Parameter references are resolved using \p call_context.
///
/// \param transaction            The DB transaction to use (to duplicate the attachments).
/// \param expr                   The expression from which to create the deep copy.
/// \param call_context           The context to resolve parameter references.
/// \return                       An copy of \p expr with all expressions replaced by duplicates.
IExpression* deep_copy(
    const IExpression_factory* ef,
    DB::Transaction* transaction,
    const IExpression* expr,
    const std::vector<mi::base::Handle<const IExpression>>& call_context);

/// Returns a hash value for a resource (light profiles and BSDF measurements).
///
/// Uses the MDL file path if not empty, and the tag version otherwise.
mi::Uint32 get_hash( const std::string& mdl_file_path, const DB::Tag_version& tv);

/// Returns a hash value for a resource (light profiles and BSDF measurements).
///
/// Uses the MDL file path if not empty, and the tag version otherwise.
mi::Uint32 get_hash( const char* mdl_file_path, const DB::Tag_version& tv);

/// Returns a hash value for resource (textures).
///
/// Uses the MDL file path and gamma value if the MDL file path is not empty, and the two tag
/// versions otherwise.
mi::Uint32 get_hash(
    const std::string& mdl_file_path,
    mi::Float32 gamma,
    const DB::Tag_version& tv1,
    const DB::Tag_version& tv2);

/// Returns a hash value for resource (textures).
///
/// Uses the MDL file path and gamma value if the MDL file path is not empty, and the two tag
/// versions otherwise.
mi::Uint32 get_hash(
    const char* mdl_file_path,
    mi::Float32 gamma,
    const DB::Tag_version& tv1,
    const DB::Tag_version& tv2);

/// Returns a reader for the given string.
mi::neuraylib::IReader* create_reader( const std::string& data);

/// Returns the minimal required MDL version for a function definition.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const Mdl_function_definition* definition);

/// Returns the minimal required MDL version for a given value.
///
/// Returns mi::mdl::IMDL::MDL_VERSION_1_0 for \c NULL arguments.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IValue* value);

/// Returns the minimal required MDL version for a given expression.
///
/// Returns mi::mdl::IMDL::MDL_VERSION_1_0 for \c NULL arguments.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IExpression* expr);

/// Returns the minimal required MDL version for a given expression list.
///
/// Returns mi::mdl::IMDL::MDL_VERSION_1_0 for \c NULL arguments.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IExpression_list* expr_list);

/// Returns the minimal required MDL version for a given annotation.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation* annotation);

/// Returns the minimal required MDL version for a given annotation block.
///
/// Returns mi::mdl::IMDL::MDL_VERSION_1_0 for \c NULL arguments.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation_block* block);

/// Returns the minimal required MDL version for a given annotation list.
///
/// Returns mi::mdl::IMDL::MDL_VERSION_1_0 for \c NULL arguments.
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, const IAnnotation_list* list);

/// Returns the minimal required MDL version for a given argument pack.
template <typename T1, typename ...T2>
mi::neuraylib::Mdl_version get_min_required_mdl_version(
    DB::Transaction* transaction, T1 arg, T2... args)
{
    return std::max( get_min_required_mdl_version( transaction, arg),
                     get_min_required_mdl_version( transaction, args...));
}

/// Helper class to maintain a call stack and check for cycles.
class Call_stack_guard
{
public:
    /// Adds \p frame to the end of the call stack.
    Call_stack_guard( std::set<DB::Tag>& call_trace, DB::Tag frame)
      : m_call_trace( call_trace),
        m_frame( frame),
        m_has_cycle( !m_call_trace.insert( m_frame).second) { }

    /// Removes the last frame from the call stack again.
    ~Call_stack_guard() { m_call_trace.erase( m_frame); }

    /// Checks whether the last frame in the call stack creates a cycle.
    bool last_frame_creates_cycle() const { return m_has_cycle; }

private:
    std::set<DB::Tag>& m_call_trace;
    DB::Tag m_frame;
    bool m_has_cycle;
};

/// Loads the neuray module in the current transaction (if not already loaded).
///
/// The side effect of this is that all standard modules including the builtins module are loaded
/// as well.
void load_neuray_module( DB::Transaction* transaction);

// **********  Traversal of types, values, and expressions *****************************************

/// Looks up a sub value according to path.
///
/// \param value           The value.
/// \param path            The path to follow in \p value. The path may contain dots or bracket
///                        pairs as separators. The path component up to the next separator is used
///                        to select struct fields by name or other compound elements by index.
/// \return                A sub-value for \p value according to \p path, or \c NULL in case of
///                        errors.
const IValue* lookup_sub_value( const IValue* value, const char* path);

/// Looks up a sub-expression according to path.
///
/// \param ef              The expression factory used to wrap values as expressions.
/// \param temporaries     Used to resolve temporary expressions.
/// \param expr            The expression. Calls and parameter references are not supported.
///                        Temporaries are only supported if \p temporaries is not \c NULL.
/// \param path            The path to follow in \p expr. The path may contain dots or bracket pairs
///                        as separators. The path component up to the next separator is used to
///                        select struct fields or direct call arguments by name or other compound
///                        elements by index.
/// \return                A sub-expression for \p expr according to \p path, or \c NULL in case of
///                        errors.
const IExpression* lookup_sub_expression(
    const IExpression_factory* ef,
    const IExpression_list* temporaries,
    const IExpression* expr,
    const char* path);

// ********** Misc utility functions around mi::mdl ************************************************

/// Converts deferred-sized arrays into immediate-sized arrays of length \p size.
const mi::mdl::IType_compound* convert_deferred_sized_into_immediate_sized_array(
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType_compound* mdl_type,
    mi::Size size);

/// Creates an MDL AST qualified name for a given function/material signature.
///
/// \param nf                     The name factory to be used.
/// \param signature              The core signature without parameter types.
/// \param name_mangler           The name mangler.
/// \return                       The MDL AST expression reference for the signature.
mi::mdl::IQualified_name* signature_to_qualified_name(
    mi::mdl::IName_factory* nf, const char* signature, Name_mangler* name_mangler);

/// Creates an MDL AST expression reference for a given function/material signature.
///
/// \param module                 The module on which the qualified name is created.
/// \param signature              The core signature without parameter types.
/// \param name_mangler           The name mangler.
/// \return                       The MDL AST expression reference for the signature.
const mi::mdl::IExpression_reference* signature_to_reference(
    mi::mdl::IModule* module, const char* signature, Name_mangler* name_mangler);

/// Helper class to associate all resource literals inside a code DAG with their DB tags.
/// Handles also bodies of called functions.
class Resource_updater  : private mi::mdl::Module_visitor
{
public:
    /// Constructor.
    ///
    /// \param transaction            The DB transaction to use.
    /// \param resolver               The call name resolver.
    /// \param code_dag               The code DAG to update.
    /// \param module_filename        The file name of the module.
    /// \param module_mdl_name        The MDL module name.
    Resource_updater(
        DB::Transaction* transaction,
        mi::mdl::ICall_name_resolver& resolver,
        mi::mdl::IGenerated_code_dag* code_dag,
        const char* module_filename,
        const char* module_mdl_name,
        Execution_context* context);

    /// Associates all resource literals inside a code DAG with their DB tags.
    void update_resource_literals();

private:
    void update_resource_literals( const mi::mdl::DAG_node* node);
    void update_resource_literals( const mi::mdl::IModule* owner, const mi::mdl::IDefinition* def);
    void update_resource_literals( const mi::mdl::IDefinition* def);
    void update_resource_literals( const mi::mdl::IValue_resource* resource);
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_literal* expr);
    mi::mdl::IExpression* post_visit( mi::mdl::IExpression_call* expr);

    DB::Transaction*              m_transaction;
    mi::mdl::ICall_name_resolver& m_resolver;
    mi::mdl::IGenerated_code_dag* m_code_dag;
    const char*                   m_module_filename;
    const char*                   m_module_mdl_name;
    Execution_context*            m_context;

    /// Keep track of visited definitions to avoid retraversal.
    using Definition_set = std::set<const mi::mdl::IDefinition*>;
    Definition_set m_definition_set;

    /// Keep track of visited resource literals to avoid re-resolving.
    using Resource_tag_map = std::map<const mi::mdl::IValue_resource*, DB::Tag>;
    Resource_tag_map m_resource_tag_map;

    /// Used to keep track of the owner module inside calls to
    /// #update_resource_literals(const mi::mdl::IModule*,...).
    const mi::mdl::IModule* m_def_owner;
};


// ********** Name_mangler *************************************************************************

/// A simple name mangler.
class Name_mangler
{
public:
    Name_mangler( mi::mdl::IMDL* mdl, mi::mdl::IModule* module);

    /// Checks that no aliases have been created that have not yet been added to the module (see
    /// #add_namespace_aliases()).
    ~Name_mangler();

    /// Returns the mangled name for a symbol.
    ///
    /// For MDL >= 1.8 no mangling takes places, and the symbol itself returned. For MDL < 1.8
    /// returns the symbol itself if no mangling is necessary. Otherwise, returns the previously
    /// created mangled name for this symbol, or creates a new one if this symbol is seen the first
    /// time. A newly mangled name is recoded as alias, and needs to be added to the module later
    /// using #add_namespace_aliases().
    const char* mangle( const char* symbol);

    /// Returns the mangled name for a scoped name.
    ///
    /// For MDL >= 1.8 no mangling takes places, and the name itself is returned. For MDL < 1.8
    /// calls mangle() on each component of the name.
    std::string mangle_scoped_name( const std::string& name);

    /// Adds the recorded aliases in \c m_to_add to the module and clears the vector.
    void add_namespace_aliases( mi::mdl::IModule* module);

private:
    /// Converts \p name into a string representation.
    static std::string stringify( const mi::mdl::IQualified_name* name);

    /// Returns a string with \p ident as prefix that is not contained in \c m_aliases.
    ///
    /// TODO This method does not check for name collisions with other identifiers in the module.
    std::string make_unique( const std::string& ident) const;

    /// Indicates whether namespace aliases are legal (up to MDL 1.7).
    bool m_namespace_aliases_legal;

    mi::base::Handle<mi::mdl::IMDL> m_mdl;

    mi::base::Handle<mi::mdl::IModule> m_module;

    /// The map of names/symbols to aliases in use for this module
    std::map<std::string, std::string> m_name_to_alias;

    /// The set of aliases in use for this module.
    std::set<std::string> m_aliases;

    /// The vector of mappings still to be added to the module.
    std::vector<std::string> m_to_add;
};

// ********** Symbol_importer **********************************************************************

class Name_importer;

/// Helper class to create imports from entities references by AST expressions.
class Symbol_importer
{
public:
    /// Constructor.
    ///
    /// \param module  The module to import into
    Symbol_importer( mi::mdl::IModule* module);

    /// Destructor.
    ~Symbol_importer();

    /// Collects all imports required by an AST expression.
    void collect_imports( const mi::mdl::IExpression* expr);

    /// Collects all imports required by a type name.
    void collect_imports(const mi::mdl::IType_name* tn);

    /// Collects all imports required by an annotation block.
    void collect_imports( const mi::mdl::IAnnotation_block* annotation_block);

    /// Add names from a list.
    void add_names( const std::set<std::string>& names);

    /// Write the collected imports into the module.
    void add_imports();

    /// Returns true if the current list of imports contains MDLE definitions.
    bool imports_mdle() const;

private:
    /// Converts \p name into a string representation.
    static std::string stringify( const mi::mdl::IQualified_name* name);

    /// The name importer.
    Name_importer* m_name_importer;

    /// Core name of this module (to avoid import loops).
    std::string m_module_core_name;

    /// Names from import declarations that exist already in the module.
    ///
    /// Used to avoid generating repeated redundant import declarations when the same module is
    /// processed several times. Only qualified import declarations without wildcard at the end are
    /// tracked (these are the ones generated by the collect_imports() methods of this class).
    std::set<std::string> m_existing_imports;
};

// ********** Misc *********************************************************************************

/// Converts a hash from the MDL API representation to the base API representation.
mi::base::Uuid convert_hash( const mi::mdl::DAG_hash& hash);

/// Returns the hash value of the resource reader (or {0,0,0,0} if not available).
mi::base::Uuid get_hash( mi::mdl::IMDL_resource_reader* reader);

/// Returns the combined hash value of all resource readers in the set (or {0,0,0,0} if not
/// available).
mi::base::Uuid get_hash( const mi::mdl::IMDL_resource_set* set);

/// Generates a unique ID.
Uint64 generate_unique_id();

/// Returns in instance of mi::IString holding the string \p s (or the empty string if \p s is
/// \c NULL).
mi::IString* create_istring( const char* s);

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_UTILITIES_H
