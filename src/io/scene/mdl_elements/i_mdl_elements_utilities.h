/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Public utilities related to MDL scene elements.

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_UTILITIES_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_UTILITIES_H

#include <any>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/neuraylib/typedefs.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/imdl_loading_wait_handle.h>
#include <mi/neuraylib/imdl_impexp_api.h>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/unordered_dense/unordered_dense.h>
#include <base/data/db/i_db_tag.h>

#include "i_mdl_elements_type.h"

namespace mi { namespace neuraylib {
class IDeserialized_module_name;
class IMdle_serialization_callback;
class IMdle_deserialization_callback;
class IReader;
class ISerialized_function_name;
} }

namespace mi { namespace mdl {
class IGenerated_code_dag;
class IInput_stream;
class IMDL;
class IMDL_resource_reader;
class IMDL_resource_set;
class IMdle_input_stream;
class IModule;
class IModule_cache;
class IModule_cache_wait_handle;
class IThread_context;
class Messages_impl;
} }

namespace MI {

namespace DB { class Transaction; }
namespace IMAGE { class IMdl_container_callback; }

namespace MDL {

class Execution_context;
class IAnnotation;
class IAnnotation_block;
class IAnnotation_list;
class IDeserialized_function_name;
class IExpression;
class IExpression_call;
class IExpression_constant;
class IExpression_direct_call;
class IExpression_factory;
class IExpression_list;
class IExpression_parameter;
class IExpression_temporary;
class IValue;
class IValue_texture;
class IValue_light_profile;
class IValue_bsdf_measurement;
class IValue_list;
class Mdl_compiled_material;
class Mdl_function_definition;
class Mdl_module;

// ********** Computation of references to other DB element ****************************************

/// Inserts all tag references in \p value into \p result.
void collect_references( const IValue* value, DB::Tag_set* result);

/// Inserts all tag references in \p list into \p result.
void collect_references( const IValue_list* list, DB::Tag_set* result);

/// Inserts all tag references in \p expr into \p result.
void collect_references( const IExpression* expr, DB::Tag_set* result);

/// Inserts all tag references in \p list into \p result.
void collect_references( const IExpression_list* list, DB::Tag_set* result);

/// Inserts all tag references in \p annotation into \p result.
void collect_references( const IAnnotation* annotation, DB::Tag_set* result);

/// Inserts all tag references in \p block into \p result.
void collect_references( const IAnnotation_block* block, DB::Tag_set* result);

/// Inserts all tag references in \p list into \p result.
void collect_references( const IAnnotation_list* list, DB::Tag_set* result);


// ********** Memory allocation helper class *******************************************************

/// A VLA using space inside the object or using an allocator if the requested size is too large.
/// \tparam T  element type, must be a POD.
/// \tparam N  preallocated size
/// \tparam A  the allocator type
template<typename T, size_t N, typename A = std::allocator<T> >
class Small_VLA {
public:
    /// Construct a new VLA of size n, using the internal space or allocated with the
    /// allocator alloc if the size is too large.
    Small_VLA(size_t n, A const &alloc = A())
        : m_data(nullptr)
        , m_size(n)
        , m_alloc(alloc)
    {
        if (m_size > N) {
            m_data = m_alloc.allocate(m_size);
        } else {
            m_data = m_static;
        }
    }

    /// Free the VLA.
    ~Small_VLA()
    {
        if (m_data != m_static)
            m_alloc.deallocate(m_data, m_size);
    }

    /// Access the VLA array by index.
    T &operator[](size_t index) {
        ASSERT(M_SCENE, index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array by index.
    T const &operator[](size_t index) const {
        ASSERT(M_SCENE, index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array.
    T *data() { return m_data; }

    /// Access the VLA array.
    T const *data() const { return m_data; }

    /// Return the size of the VLA.
    size_t size() const { return m_size; }

private:
    T *m_data;
    T m_static[N];
    size_t const m_size;

    A m_alloc;
};

// ********** Name parsing/splitting ***************************************************************

// These functions are supposed to centralize all parsing/splitting of strings.
//
// Many other functions are declared in the internal header file mdl_elements_utilites.h, but could
// be moved to this interface header file if necessary.

/// Indicates whether \p name is valid a MDL module name (does not support MDLE).
bool is_valid_module_name( const std::string& name);

/// Indicates whether the module or definition name is from an MDLE module. MDL or DB names.
///
/// Checks for ".mdle" suffix or ".mdle::" substring.
bool is_mdle( const std::string& name);

/// Indicates whether \p name starts with "::".
bool starts_with_scope( const std::string& name);

/// Normalizes an argument to mi::neuraylib::IMdl_impexp_api::load_module().
///
/// Takes a module identifier, normalized it for MDLEs, and return the MDL module name.
///
/// For non-MDLE modules, the method adds the prefix "::" if there is none yet, i.e., the method
/// returns the MDL name. (Non-MDLE module names without leading "::" are deprecated.)
///
/// For MDLE module names, the method
/// - makes non-absolute path absolute w.r.t. the current working directory,
/// - normalizes them (handling of directory components "." and ".."),
/// - converts them to use forward slashes,
/// - adds a slash in front of drive letter (Windows only)
/// - adds a scope ("::") prefix
/// - and finally encodes them.
std::string get_mdl_name_from_load_module_arg( const std::string& name, bool is_mdle);

/// Converts an MDL name of a module, material or function definition into the corresponding
/// DB name.
///
/// Does not work for annotation definitions, #use get_db_name_annotation_definition() instead.
///
/// For non-MDLE names, it adds the prefix "mdl" (or "mdl::" if \p name does not start with "::").
///
/// For MDLE names, it adds the prefix "mdle".
std::string get_db_name( const std::string& name);

/// Removes owner (module) prefix from resource URLs.
///
/// Resource URLs without '::' are returned as is.
std::string strip_resource_owner_prefix( const std::string& name);

/// Returns the owner (module) prefix from resource URLs.
///
/// Returns the empty string for resource URLs without '::'.
std::string get_resource_owner_prefix( const std::string& name);

// ********** Misc utility functions ***************************************************************

/// Returns the DB element name used for the array constructor.
const char* get_array_constructor_db_name();

/// Returns the MDL name used for the array constructor.
const char* get_array_constructor_mdl_name();

/// Returns the DB name used for the index operator.
const char* get_index_operator_db_name();

/// Returns the MDL name used for the index operator.
const char* get_index_operator_mdl_name();

/// Returns the DB name used for the array length operator.
const char* get_array_length_operator_db_name();

/// Returns the MDL name used for the array length operator.
const char* get_array_length_operator_mdl_name();

/// Returns the DB name used for the ternary operator.
const char* get_ternary_operator_db_name();

/// Returns the MDL name used for the ternary operator.
const char* get_ternary_operator_mdl_name();

/// Returns the DB element name used for the cast operator.
const char* get_cast_operator_db_name();

/// Returns the MDL name used for the cast operator.
const char* get_cast_operator_mdl_name();

/// Returns the DB element name used for the decl_cast operator.
const char* get_decl_cast_operator_db_name();

/// Returns the MDL name used for the decl_cast operator.
const char* get_decl_cast_operator_mdl_name();

/// Returns the DB name used for the builtins module ("mdl::%3Cbuiltins%3E").
const char* get_builtins_module_db_name();

/// Returns the MDL name used for the builtins module ("::%3Cbuiltins%3E").
const char* get_builtins_module_mdl_name();

/// Returns the simple name used for the builtins module ("%3Cbuiltins%3E").
const char* get_builtins_module_simple_name();

/// Returns the DB name used for the neuray module ("mdl::%3Cneuray%3E").
const char* get_neuray_module_db_name();

/// Returns the MDL name used for the neuray module ("::%3Cneuray%3E").
const char* get_neuray_module_mdl_name();

/// Encodes a string with percent-encoding.
///
/// Encoded characters are parentheses, angle brackets, comma, colon, dollar, hash, question mark,
/// at sign, and percent.
std::string encode( const char* s);

/// Decodes a string with percent-encoding.
///
/// Hexadecimal digits need to be upper-case. Exactly the characters mentioned for #encode() must
/// be encoded. Returns the empty string in case of errors.
///
/// If \p strict is \c false, then encoding errors are ignored (useful for error messages, but
/// otherwise indicates an earlier mistake).
std::string decode(
    const std::string& s, bool strict = true, Execution_context* context = nullptr);

/// Decodes a string with percent-encoding for error messages.
///
/// In error messages, we do not care about encoding errors and want to emit as much information
/// as possible. Calls #decode() with \c false for \c strict.
std::string decode_for_error_msg( const std::string& s);

/// Encodes a module name (no signature, no $ suffix).
///
/// Calls #encode() on components delimited by "::".
std::string encode_module_name( const std::string& s);

/// Decodes a module name (no signature, no $ suffix).
///
/// Calls #decode() on components delimited by "::".
/// Returns the empty string in case of errors.
std::string decode_module_name( const std::string& s);

/// Encodes a function, material, annotation, or type name (no signature, possibly $ suffix).
///
/// Calls #encode() on components delimited by "::" (last component might contain "$").
std::string encode_name_without_signature( const std::string& s);

/// Decodes a function, material, annotation, or type name (no signature, possibly $ suffix).
///
/// Calls #decode() on components delimited by "::" (last component might contain "$").
/// Returns the empty string in case of errors.
std::string decode_name_without_signature( const std::string& s);

/// Encodes a name with signature.
///
/// \note This function assumes that parentheses, comma, and dollar are always meta-characters and
///       do not appear as part of a simple name. TODO encoded names: Remove this limitation.
std::string encode_name_with_signature( const std::string& s);

/// Decodes a name with signature.
///
/// \note This function assumes that parentheses, comma, and dollar are always meta-characters and
///       do not appear as part of a simple name. TODO encoded names: Remove this limitation.
///
/// Does not reject invalid input.
std::string decode_name_with_signature( const std::string& s);

/// Encodes a name plus signature.
std::string encode_name_plus_signature(
    const std::string& s, const std::vector<std::string>& parameter_types);

/// Returns an encoded material or function name with signature.
///
/// Calls #encode() on components delimited by parentheses and "::". Used to translate
/// mi::mdl::IGenerated_code_dag.
///
/// \note This function assumes that parentheses are always meta-characters and do not appear as
///       part of a simple name. TODO encoded names: Remove this limitation.
std::string get_mdl_name(
    const mi::mdl::IGenerated_code_dag* code_dag, bool is_material, mi::Size index);

/// Returns an encoded annotation name with signature.
///
/// Calls #encode() on components delimited by parentheses and"::". Used to translate
/// mi::mdl::IGenerated_code_dag.
///
/// \note This function assumes that parentheses are always meta-characters and do not appear as
///       part of a simple name. TODO encoded names: Remove this limitation.
std::string get_mdl_annotation_name(
    const mi::mdl::IGenerated_code_dag* code_dag, mi::Size index);

/// Encodes a string with or without signature (and adds a missing signature for materials).
///
/// Tries to find the name in the code DAG (if not \c nullptr), and uses the DB as fallback.
///
/// \note This function assumes that parentheses, comma, and dollar are always meta-characters and
///       do not appear as part of a simple name. TODO encoded names: Remove this limitation.
std::string encode_name_add_missing_signature(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag* m_code_dag,
    const std::string& name);

/// Returns the serialized function name for the given function definition name.
///
/// \note This method re-uses in its return value the interface from the public API since it is
///       trivial to implement and an internal variant would look identical.
///
/// \param definition_name   The DB name of the corresponding function of material definition. This
///                          name is not checked for correctness or presence in the DB.
/// \param argument_types    The actual argument types of the function call. Required for
///                          template-like functions, ignored otherwise.
/// \param return_type       The actual return type of the function call. Required for the cast
///                          operator, ignored otherwise.
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c nullptr (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The serialized function definition (and module) name, or \c nullptr in
///                          case of errors.
const mi::neuraylib::ISerialized_function_name* serialize_function_name(
    const char* definition_name,
    const IType_list* argument_types,
    const IType* return_type,
    mi::neuraylib::IMdle_serialization_callback* mdle_callback,
    Execution_context* context);

/// Returns the deserialized function definition name for the given serialized function name.
///
/// Loads the module if necessary. Performs overload resolution.
///
/// \param transaction       The DB transaction to use.
/// \param function_name     The serialized function name.
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c nullptr (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized function name and argument types, or \c nullptr in
///                          case of errors.
const IDeserialized_function_name* deserialize_function_name(
    DB::Transaction* transaction,
    const char* function_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context);

/// Returns the deserialized function definition name for the given pair of serialized module name
/// and function name without module name.
///
/// Loads the module if necessary. Performs overload resolution.
///
/// \param transaction       The DB transaction to use.
/// \param module_name       The serialized module name.
/// \param function_name_without_module_name   The serialized function name without module name.
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c nullptr (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized function name and argument types, or \c nullptr in
///                          case of errors.
const IDeserialized_function_name* deserialize_function_name(
    DB::Transaction* transaction,
    const char* module_name,
    const char* function_name_without_module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context);

/// Returns the deserialized module name for the given serialized module name.
///
/// Loads the module if necessary. Performs overload resolution.
///
/// \param module_name       The serialized module name.
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c nullptr (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized module name, or \c nullptr in case of errors.
const mi::neuraylib::IDeserialized_module_name* deserialize_module_name(
    const char* module_name,
    mi::neuraylib::IMdle_deserialization_callback* mdle_callback,
    Execution_context* context);

/// Internal variant of #mi::neuraylib::IDeserialized_function_name (uses MDL::IType_list instead of
/// mi::neuraylib::IType_list).
class IDeserialized_function_name : public
    mi::base::Interface_declare<0x6afc41ef,0x9adf,0x446a,0x80,0x21,0x8f,0x62,0x6c,0x77,0x24,0xb4>
{
public:
    /// Returns the DB name.
    virtual const char* get_db_name() const = 0;

    /// Returns the corresponding argument types suitable for
    /// #Mdl_function_definition::create_call(), e.g., two argument types for the cast operator.
    virtual const IType_list* get_argument_types() const = 0;
};

/// Returns \c true for builtin modules.
///
/// Builtin modules have no underlying file, e.g., "::state".
///
/// \param module   The core name of the module (including leading double colons, e.g., "::state").
bool is_builtin_module( const std::string& module);

/// Loads the builtin mdl module ::nvidia::distilling_support into the given transaction.
///
/// \param transaction  the transaction
void load_distilling_support_module( DB::Transaction* transaction);

/// Indicates whether a function of the given semantic is a valid prototype for prototype-based
/// functions or variants.
bool is_supported_prototype(
    mi::neuraylib::IFunction_definition::Semantics sema, bool for_variant);

/// Returns a default compiled material.
///
/// Can be used as a default for failed compilation steps. Returns \c nullptr if someone explicitly
/// removed the corresponding material definition from the DB without removing the entire module
/// (should not happen).
Mdl_compiled_material* get_default_compiled_material( DB::Transaction* transaction);

/// Returns the MDL module name of the module which defines \p type.
///
/// The module name is derived from the symbol of enum, struct, and alias types. For array
/// types, the corresponding element type is considered. Frequency modifiers of alias types are
/// ignored. Returns the name of the \c ::&lt;builtins&gt; module for all other types.
std::string get_mdl_module_name( const IType* type);

/// Represents an MDL compiler message. Similar to mi::mdl::IMessage.
class Message
{
public:
    enum Kind {
        MSG_COMPILER_CORE,
        MSG_COMPILER_BACKEND,
        MSG_COMPILER_DAG,
        MSG_COMPILER_ARCHIVE_TOOL,
        MSG_IMP_EXP,
        MSG_INTEGRATION,
        MSG_UNCATEGORIZED,
    };

    Message(
        mi::base::Message_severity severity, const std::string& message)
      : m_severity( severity),
        m_message( message) { }

    Message(
        mi::base::Message_severity severity, const std::string& message, mi::Sint32 code, Kind kind)
      : m_severity( severity),
        m_code( code),
        m_message( message),
        m_kind( kind) { }

    Message( const mi::mdl::IMessage*message);

    mi::base::Message_severity  m_severity;
    mi::Sint32                  m_code = -1;
    std::string                 m_message;
    Kind                        m_kind = MSG_UNCATEGORIZED;
    std::vector<Message>        m_notes;
};

/// Simple Option class.
class Option
{
public:

    using Validator = bool (*)(const std::any&);

    // Default constructor. Required by the hash map.
    Option() = default;

    // Regular constructor.
    Option( const std::any& default_value, bool is_interface, Validator validator = nullptr)
      : m_value( default_value),
        m_validator( validator),
        m_is_interface( is_interface)
    { }

    bool set_value( const std::any& value)
    {
        if( m_validator && !m_validator( value))
            return false;
        m_value = value;
        return true;
    }

    const std::any& get_value() const { return m_value; }

    bool is_interface() const { return m_is_interface; }

private:

    std::any m_value;
    Validator m_validator = nullptr;
    bool m_is_interface = false;
};

// When adding new options
// - adapt the Execution_context constructor to register the default and its type
// - adapt create_thread_context() and create_execution_context() if necessary

#define MDL_CTX_OPTION_WARNING                             "warning"
#define MDL_CTX_OPTION_OPTIMIZATION_LEVEL                  "optimization_level"
#define MDL_CTX_OPTION_INTERNAL_SPACE                      "internal_space"
#define MDL_CTX_OPTION_FOLD_METERS_PER_SCENE_UNIT          "fold_meters_per_scene_unit"
#define MDL_CTX_OPTION_METERS_PER_SCENE_UNIT               "meters_per_scene_unit"
#define MDL_CTX_OPTION_WAVELENGTH_MIN                      "wavelength_min"
#define MDL_CTX_OPTION_WAVELENGTH_MAX                      "wavelength_max"
#define MDL_CTX_OPTION_INCLUDE_GEO_NORMAL                  "include_geometry_normal"
#define MDL_CTX_OPTION_BUNDLE_RESOURCES                    "bundle_resources"
#define MDL_CTX_OPTION_EXPORT_RESOURCES_WITH_MODULE_PREFIX "export_resources_with_module_prefix"
#define MDL_CTX_OPTION_HANDLE_FILENAME_CONFLICTS           "handle_filename_conflicts"
#define MDL_CTX_OPTION_FILENAME_HINTS                      "filename_hints"
#define MDL_CTX_OPTION_MDL_NEXT                            "mdl_next"
#define MDL_CTX_OPTION_EXPERIMENTAL                        "experimental"
#define MDL_CTX_OPTION_RESOLVE_RESOURCES                   "resolve_resources"
#define MDL_CTX_OPTION_FOLD_TERNARY_ON_DF                  "fold_ternary_on_df"
#define MDL_CTX_OPTION_IGNORE_NOINLINE                     "ignore_noinline"
#define MDL_CTX_OPTION_REMOVE_DEAD_PARAMETERS              "remove_dead_parameters"
#define MDL_CTX_OPTION_FOLD_ALL_BOOL_PARAMETERS            "fold_all_bool_parameters"
#define MDL_CTX_OPTION_FOLD_ALL_ENUM_PARAMETERS            "fold_all_enum_parameters"
#define MDL_CTX_OPTION_FOLD_PARAMETERS                     "fold_parameters"
#define MDL_CTX_OPTION_FOLD_TRIVIAL_CUTOUT_OPACITY         "fold_trivial_cutout_opacity"
#define MDL_CTX_OPTION_FOLD_TRANSPARENT_LAYERS             "fold_transparent_layers"
#define MDL_CTX_OPTION_SERIALIZE_CLASS_INSTANCE_DATA       "serialize_class_instance_data"
#define MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY         "loading_wait_handle_factory"
#define MDL_CTX_OPTION_DEPRECATED_REPLACE_EXISTING         "replace_existing"
#define MDL_CTX_OPTION_TARGET_MATERIAL_MODEL_MODE          "target_material_model_mode"
#define MDL_CTX_OPTION_USER_DATA                           "user_data"
#define MDL_CTX_OPTION_TARGET_TYPE                         "target_type"
// Not documented in the API (used by the module transformer, but not for general use).
#define MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS   "keep_original_resource_file_paths"

/// Represents an MDL execution context, similar to mi::mdl::Thread_context.
///
/// There is one instance holding all the defaults, and all other instances hold the explicitly
/// set options (whose values might be identical to the default). Without the explicit instance
/// with all the defaults, the time to create all the defaults for all temporaries becomes
/// noticeable in benchmarks.
class Execution_context
{
public:

    Execution_context() : Execution_context( /*add_defaults*/ false) { }

    // Error messages

    mi::Size get_messages_count() const;

    mi::Size get_error_messages_count() const;

    const Message& get_message( mi::Size index) const;

    const Message& get_error_message( mi::Size index) const;

    void add_message( const mi::mdl::IMessage* message);

    void add_error_message( const mi::mdl::IMessage* message);

    void add_message( const Message& message);

    void add_error_message( const Message& message);

    void add_messages( const mi::mdl::Messages& messages);

    void clear_messages();

    /// The result is purely internal and not exposed in the API.
    ///
    /// It is meant to store a result code for functions with mi::Sint32 return type, either in
    /// addition to an execution context, or without execution context. The following invariants
    /// should be maintained:
    /// - Result is zero => no error messages
    /// - Result is non-zero => at least one error message
    /// In case of non-zero result, there is typically also an error message with the same code.
    /// But this is not necessarily the case if messages from the core compiler are passed and no
    /// overall message for the result code is generated.
    void set_result( mi::Sint32 result);

    mi::Sint32 get_result() const;

    // Options

    mi::Size get_option_count() const;

    const char* get_option_name( mi::Size index) const;

    mi::Sint32 get_option( const std::string& name, std::any& value) const;

    mi::Sint32 set_option( const std::string& name, const std::any& value);

    const Option* get_option( const std::string& name) const;

    /// The option's value type has to be T, otherwise an exception might be thrown.
    template<typename T>
    T get_option( const std::string& name) const
    {
        const Option* option = get_option( name);
        ASSERT( M_SCENE, option);
        ASSERT( M_SCENE, !option->is_interface());

        return std::any_cast<T>( option->get_value());
    }

    /// The option's value type has to be mi::base::Handle<T>, otherwise an exception might be
    /// thrown.
    template<typename T>
    T* get_interface_option( const std::string& name) const
    {
        const Option* option = get_option( name);
        ASSERT( M_SCENE, option);
        ASSERT( M_SCENE, option->is_interface());

        mi::base::Handle<const mi::base::IInterface> handle(
            std::any_cast<mi::base::Handle<const mi::base::IInterface>>( option->get_value()));
        if( !handle)
            return nullptr;
        return handle->get_interface<T>();
    }

private:
    Execution_context( bool add_defaults);

    void add_default_option( const char* name, const Option& option);

    std::vector<Message> m_messages;
    std::vector<Message> m_error_messages;
    mi::Sint32 m_result = 0;

    /// Points to an instance holding the default options (or \c nullptr if this is the instance
    /// holding the default options).
    const Execution_context* m_default_options = nullptr;

    /// The currently set options (or all default options, see \c m_default_options).
    ankerl::unordered_dense::map<std::string, Option> m_options;

    /// Names of all default options (only valid for the instance holding the default options).
    std::vector<std::string> m_names;

};

/// Adds core messages to an execution context.
void convert_messages( const mi::mdl::Messages& in_messages, Execution_context* context);

/// Adds messages from an execution context to the core messages.
void convert_messages( const Execution_context* context, mi::mdl::Messages& out_messages);

/// Logs the messages in an execution context.
void log_messages( const Execution_context* context, mi::Size start_index);

/// Logs the core messages.
void log_messages( const mi::mdl::Messages& messages);

/// Adds core messages to an optional execution context and logs them.
void convert_and_log_messages( const mi::mdl::Messages& in_messages, Execution_context* context);

/// Adds \p message as message to the context.
///
/// If the severity is #mi::base::MESSAGE_SEVERITY_ERROR or #mi::base::MESSAGE_SEVERITY_FATAL, then
/// the message is also added as error message to the context, and the context result is set to
/// \p result.
///
/// Does nothing if \p context is \c nullptr. Returns \p result.
mi::Sint32 add_message( Execution_context* context, const Message& message, mi::Sint32 result);

/// Adds \p message as message and error message to the context, and sets the result to
/// \p result_and_code.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and \p result_and_code as message code.
/// Does nothing if \p context is \c nullptr. Returns \p result_and_code.
mi::Sint32 add_error_message(
    Execution_context* context, const std::string& message, mi::Sint32 result_and_code);

/// Adds \p message as warning message to the context.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and -1 as message code.
/// Does nothing if \p context is \c nullptr.
void add_warning_message( Execution_context* context, const std::string& message);

/// Adds \p message as info message to the context.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and -1 as message code.
/// Does nothing if \p context is \c nullptr.
void add_info_message( Execution_context* context, const std::string& message);

/// Creates a thread context.
///
/// If \p context is not \c nullptr, its relevant options are copied to the thread context. (Not all
/// API context options are passed to the core compiler via the thread context, hence, only a
/// subset, the relevant ones, are copied by this method.)
mi::mdl::IThread_context* create_thread_context( mi::mdl::IMDL* mdl, Execution_context* context);

/// Creates a thread context.
///
/// If \p context is not \c nullptr, its relevant options are copied to the thread context. (Not all
/// API context options are passed to the core compiler via the thread context, hence, only a
/// subset, the relevant ones, are copied by this method.)
///
/// Convenience wrapper for the method above. Obtains the mi::mdl::IMDL instance from
/// Access_module<MDLC::Mdlc_module>.
mi::mdl::IThread_context* create_thread_context( Execution_context* context);

/// Creates an execution context.
///
/// If \p ctx is not \c nullptr, its relevant options are copied to the execution context. (Not all
/// API context options are passed to the core compiler via the thread context, hence, only a
/// subset, the relevant ones, are copied by this method.)
Execution_context* create_execution_context( mi::mdl::IThread_context* ctx);


// ********** Readers, writers & streams ***********************************************************

/// An mi::mdl::IOutput_stream with the extra functionality to check for errors.
class IOutput_stream : public mi::base::Interface_declare<
    0xb3f136fb,0xba77,0x4ec6,0x9e,0xe6,0x95,0x18,0xdf,0x4a,0xb2,0x6d, mi::mdl::IOutput_stream>
{
public:
    /// Indicates whether the stream is in an error state.
    virtual bool has_error() const = 0;
};

/// Wraps an core input stream as IReader.
mi::neuraylib::IReader* get_reader( mi::mdl::IInput_stream* stream);

/// Wraps an core resource reader as IReader.
mi::neuraylib::IReader* get_reader( mi::mdl::IMDL_resource_reader* resource_reader);

/// Wraps an IReader as core input stream.
mi::mdl::IInput_stream* get_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename);

/// Wraps an IWriter as core output stream.
MDL::IOutput_stream* get_output_stream( mi::neuraylib::IWriter* writer);

/// Wraps an IReader as core MDLE input stream.
mi::mdl::IMdle_input_stream* get_mdle_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename);

/// Wraps an IReader as core resource reader. The reader needs to support absolute access.
mi::mdl::IMDL_resource_reader* get_resource_reader(
    mi::neuraylib::IReader* reader,
    const std::string& file_path,
    const std::string& filename,
    const mi::base::Uuid& hash);

/// Returns a reader to a resource from an MDL archive or MDLE file.
mi::neuraylib::IReader* get_container_resource_reader(
    const std::string& resolved_container_filename,
    const std::string& container_member_name);

/// Creates an instance of an implementation of IMAGE::IMdl_container_callback.
IMAGE::IMdl_container_callback* create_mdl_container_callback();


// ********** Resource-related attributes **********************************************************

/// Retrieve the attributes of a texture resource (general).
///
/// \param transaction        The DB transaction to use.
/// \param tex_tag            A texture tag.
/// \param[out] valid         The result of texture_isvalid().
/// \param[out] first_frame   The first frame of of the texture.
/// \param[out] last_frame    The last frame of of the texture.
void get_texture_attributes(
    DB::Transaction* transaction,
    DB::Tag tex_tag,
    bool& valid,
    int& first_frame,
    int& last_frame);

/// Retrieve the attributes of a texture resource (per frame/uvtile).
///
/// \param transaction        The DB transaction to use.
/// \param tex_tag            A texture tag.
/// \param frame_number       The frame number.
/// \param uvtile_u           The u-coordinate of the uvtile (or 0 for non-uvtiles).
/// \param uvtile_v           The v-coordinate of the uvtile (or 0 for non-uvtiles).
/// \param[out] valid         The result of texture_isvalid().
/// \param[out] width         The width of the texture.
/// \param[out] height        The height of the texture.
/// \param[out] depth         The depth of the texture.
/// \param[out] first_frame   The first frame of of the texture.
/// \param[out] last_frame    The last frame of of the texture.
void get_texture_attributes(
    DB::Transaction* transaction,
    DB::Tag tex_tag,
    mi::Size frame_number,
    mi::Sint32 uvtile_u,
    mi::Sint32 uvtile_v,
    bool& valid,
    int& width,
    int& height,
    int& depth,
    int& first_frame,
    int& last_frame);

/// Retrieve the attributes of a light profile resource.
///
/// \param transaction       The DB transaction to use.
/// \param lp_tage           A light profile tag.
/// \param[out] valid        The result of light_profile_isvalid().
/// \param[out] power        The result of light_profile_power().
/// \param[out] maximum      The result of light_profile_maximum().
void get_light_profile_attributes(
    DB::Transaction* transaction,
    DB::Tag lp_tag,
    bool& valid,
    float& power,
    float& maximum);

/// Retrieve the attributes of a BSDF measurement resource.
///
/// \param transaction       The DB transaction to use.
/// \param bm_tag            A BSDF measurement tag.
/// \param[out] valid        The result of bsdf_measurement_isvalid().
void get_bsdf_measurement_attributes(
    DB::Transaction* transaction,
    DB::Tag bm_tag,
    bool& valid);


// ********** Conversion from MI::MDL to mi::mdl ***************************************************

/// Converts MI::MDL::IStruct_category to mi::mdl::IStruct_category.
const mi::mdl::IStruct_category* int_struct_category_to_core_struct_category(
    const IStruct_category* struct_category,
    mi::mdl::IType_factory& tf);

/// Converts MI::MDL::IType to mi::mdl::IType.
const mi::mdl::IType* int_type_to_core_type(
    const IType* type,
    mi::mdl::IType_factory& tf);

/// Converts MI::MDL::IValue to mi::mdl::IValue.
const mi::mdl::IValue* int_value_to_core_value(
    DB::Transaction* transaction,
    mi::mdl::IValue_factory* vf,
    const mi::mdl::IType* core_type,
    const IValue* value);

/// Converts a path from the SDK representation (dots and array index brackets) to the MDL core
/// representation (dots only).
///
/// The simple search-and-replace fails to reject some invalid input like "foo[bar".
std::string int_path_to_core_path( const char* path);

// ********** Code_dag *****************************************************************************

/// Wrapper around mi::mdl::IGenerated_code_dag that dispatches between functions and materials.
class Code_dag
{
public:
    Code_dag( const mi::mdl::IGenerated_code_dag* code_dag, bool is_material)
      : m_code_dag( code_dag), m_is_material( is_material) { }

    /// Names (cloned and original might be nullptr)
    const char* get_name( mi::Size index) const;
    const char* get_simple_name( mi::Size index) const;
    const char* get_cloned_name( mi::Size index) const;
    const char* get_original_name( mi::Size index) const;

    /// Properties
    mi::mdl::IDefinition::Semantics get_semantics( mi::Size index) const;
    bool get_exported( mi::Size index) const;
    bool get_declarative( mi::Size index) const;
    bool get_uniform( mi::Size index) const;

    /// Return type
    const mi::mdl::IType* get_return_type( mi::Size index) const;

    /// Parameters
    mi::Size get_parameter_count( mi::Size index) const;
    const mi::mdl::IType* get_parameter_type( mi::Size index, mi::Size parameter_index) const;
    const char* get_parameter_type_name( mi::Size index, mi::Size parameter_index) const;
    const char* get_parameter_name( mi::Size index, mi::Size parameter_index) const;
    mi::Size get_parameter_index( mi::Size index, const char* parameter_name) const;
    const mi::mdl::DAG_node* get_parameter_default( mi::Size index, mi::Size parameter_index) const;

    /// Parameters enabled_if()
    const mi::mdl::DAG_node* get_parameter_enable_if_condition(
        mi::Size index, mi::Size parameter_index) const;
    mi::Size get_parameter_enable_if_condition_users(
        mi::Size index, mi::Size parameter_index) const;
    mi::Size get_parameter_enable_if_condition_user(
        mi::Size index, mi::Size parameter_index, mi::Size user_index) const;

    /// Parameter annotations
    mi::Size get_parameter_annotation_count( mi::Size index, mi::Size parameter_index) const;
    const mi::mdl::DAG_node* get_parameter_annotation(
        mi::Size index, mi::Size parameter_index, mi::Size annotation_index) const;

    /// Temporaries
    mi::Size get_temporary_count( mi::Size index) const;
    const mi::mdl::DAG_node* get_temporary( mi::Size index, mi::Size temporary_index) const;
    const char* get_temporary_name( mi::Size index, mi::Size temporary_index) const;

    /// Annotations
    mi::Size get_annotation_count( mi::Size index) const;
    const mi::mdl::DAG_node* get_annotation( mi::Size index, mi::Size annotation_index) const;
    mi::Size get_return_annotation_count( mi::Size index) const;
    const mi::mdl::DAG_node* get_return_annotation(
        mi::Size index, mi::Size annotation_index) const;

    /// Body
    const mi::mdl::DAG_node* get_body( mi::Size index) const;

    /// Hash (might be nullptr)
    const mi::mdl::DAG_hash* get_hash( mi::Size index) const;

private:
    const mi::mdl::IGenerated_code_dag* const m_code_dag;
    const bool m_is_material;
};

// ********** Dag_cloner ***************************************************************************

/// Helper class to import DAG nodes into another DAG builder.
///
/// Unfortunately, mi::mdl::IDag_builder does not export such a method.
class Dag_importer
{
public:
    /// Constructor.
    ///
    /// \param dag_builder   The DAG builder used to construct the DAG nodes.
    Dag_importer( mi::mdl::IDag_builder* dag_builder);

    /// Destructor.
    ~Dag_importer() { m_dag_builder->enable_opt( m_enable_opt); }

    /// Imports a DAG node from a different DAG factory into this DAG builder.
    const mi::mdl::DAG_node* import( const mi::mdl::DAG_node* node);

    /// Imports a value from a different value factory into this DAG builder.
    const mi::mdl::IValue* import( const mi::mdl::IValue* value);

    /// Imports a type from a different value factory into this DAG builder.
    const mi::mdl::IType* import( const mi::mdl::IType* value);

    /// Disable optimizations on DAG_node construction and return the old value.
    bool enable_opt( bool flag) { return m_dag_builder->enable_opt( flag); }

    /// Returns the cached converted temporaries.
    const std::vector<const mi::mdl::DAG_node*>& get_temporaries() const
    { return m_temporaries; }

protected:
    /// Returns \c nullptr (explicit method for simpler debugging).
    const mi::mdl::DAG_node* error_node();

    /// The DAG builder used to construct the DAG nodes.
    mi::mdl::IDag_builder* m_dag_builder;
    /// The type factory of the DAG builder.
    mi::mdl::IType_factory* m_type_factory;
    /// The value factory of the DAG builder.
    mi::mdl::IValue_factory* m_value_factory;

    /// Original setting of the optimize flag.
    bool m_enable_opt;

    /// The converted temporaries.
    std::vector<const mi::mdl::DAG_node*> m_temporaries;
};

// ********** Mdl_dag_builder **********************************************************************

/// Helper class to handle conversion from MDL::IExpression into mi::mdl::DAG_node.
class Mdl_dag_builder : public Dag_importer
{
public:
    /// Constructor.
    ///
    /// \param transaction         The DB transaction to use (needed to access attached function
    ///                            calls or material instances).
    /// \param dag_builder         A DAG builder used to construct the DAG nodes.
    /// \param compiled_material   The compiled material that will be used to resolve temporaries.
    ///                            Can be \c nullptr if the expressions to be converted do not
    ///                            contain any references to temporaries.
    Mdl_dag_builder(
        DB::Transaction* transaction,
        mi::mdl::IDag_builder* dag_builder,
        const Mdl_compiled_material* compiled_material);

    /// Converts an MDL::IExpression plus mi::mdl::IType into an mi::mdl::DAG_node.
    ///
    /// \note This method is meant for external callers. Do \em not use this method from within the
    ///       class in recursive calls since it lacks the \c call_stack parameter and thus using it
    ///       in such contexts breaks the cycle detection.
    ///
    /// \note If T == IGenerated_code_dag::DAG_node_factory, the converted temporaries are cached.
    ///       Multiple calls to this method must not be made, unless the temporary indices among all
    ///       expressions refer to the same vector of temporaries.
    ///
    /// \param core_type                   The core type corresponding to \p expr.
    /// \param expr                        The expression to convert.
    /// \return                            The created MDL DAG node, or \c nullptr in case of
    ///                                    failures, e.g., argument type mismatches (including
    ///                                    further down the call graph).
    const mi::mdl::DAG_node* int_expr_to_core_dag_node(
        const mi::mdl::IType* core_type, const IExpression* expr);

    /// Returns the cached parameter types.
    const std::vector<const mi::mdl::IType*>& get_parameter_types() const
    { return m_parameter_types; }

private:
    const mi::mdl::DAG_node* int_expr_constant_to_core_dag_node(
        const mi::mdl::IType* core_type,
        const IExpression_constant* expr);

    const mi::mdl::DAG_node* int_expr_call_to_core_dag_node(
        const mi::mdl::IType* core_type,
        const IExpression_call* expr);

    const mi::mdl::DAG_node* int_expr_direct_call_to_core_dag_node(
        const mi::mdl::IType* core_type,
        const IExpression_direct_call* expr);

    const mi::mdl::DAG_node* int_expr_parameter_to_core_dag_node(
        const mi::mdl::IType* core_type,
        const IExpression_parameter* expr);

    const mi::mdl::DAG_node* int_expr_temporary_to_core_dag_node(
        const mi::mdl::IType* core_type,
        const IExpression_temporary* expr);

    /// Shared between #int_expr_call_to_core_dag_node() and
    /// #int_expr_direct_call_to_core_dag_node().
    ///
    /// \param call_name    DB name of the corresponding function definition.
    const mi::mdl::DAG_node* int_expr_call_to_core_dag_node_shared(
        const mi::mdl::IType* core_type,
        const Mdl_module* module,
        bool is_material,
        mi::Size definition_index,
        DB::Tag call_tag,
        const IExpression_list* arguments);

    /// Adds \p value to m_converted_call_expressions and returns it.
    const mi::mdl::DAG_node* add_cache_entry( DB::Tag tag, const mi::mdl::DAG_node* node);

    /// The DB transaction to use (needed to access attached function calls or material instances).
    DB::Transaction* m_transaction;
    /// The core material instances used to resolve temporary references.
    mi::base::Handle<const mi::mdl::IMaterial_instance> m_core_material_instance;
    /// The core types of the parameter references.
    std::vector<const mi::mdl::IType*> m_parameter_types;
    /// Set of indirect calls in the current call stack, used to check for cycles.
    ankerl::unordered_dense::set<DB::Tag> m_set_indirect_calls;
    /// Cache of already converted function calls or material instances.
    std::map<DB::Tag, const mi::mdl::DAG_node*> m_converted_call_expressions;
};


// ********** Mdl_call_resolver ********************************************************************

/// Finds the owning module of entities.
class Mdl_call_resolver : public mi::mdl::ICall_name_resolver
{
public:
    /// Constructor.
    ///
    /// \param transaction  the transaction to use for name lookup
    Mdl_call_resolver(DB::Transaction* transaction) : m_transaction( transaction) { }

    virtual ~Mdl_call_resolver();

    /// Find the owner module of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins module,
    /// which you can identify by IModule::is_builtins().
    ///
    /// \param entity_name    the entity name (note: this cannot be a module name)
    ///
    /// \returns the owning module of this entity if found, \c nullptr otherwise
    const mi::mdl::IModule* get_owner_module(const char* name) const override;

    /// Find the owner code DAG of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins DAG,
    /// which you can identify by calling its owner module's IModule::is_builtins().
    ///
    /// \param entity_name    the entity name (note: this cannot be a module name)
    ///
    /// \returns the owning module of this entity if found, \c nullptr otherwise
    const mi::mdl::IGenerated_code_dag* get_owner_dag(const char* entity_name) const override;

private:
    DB::Tag get_module_tag(const char* entity_name) const;

    DB::Transaction* m_transaction;

    using Module_set = std::set<const mi::mdl::IModule*>;
    mutable Module_set m_resolved_modules;
};

/// Extended version of Mdl_call_resolver: looks first in a passed Module (which might not
/// be in the data base yet).
class Mdl_call_resolver_ext : public Mdl_call_resolver
{
    using Base = Mdl_call_resolver;

public:
    /// Constructor.
    ///
    /// \param transaction  the transaction to use for name lookup
    Mdl_call_resolver_ext(
        DB::Transaction* transaction,
        const mi::mdl::IModule* module);

    /// Finds the owning module of a function definition.
    ///
    /// \param name   The core name of a function definition.
    /// \return       The owning module, or \c nullptr in case of failures.
    const mi::mdl::IModule* get_owner_module(const char* name) const override;

private:
    const mi::mdl::IModule* m_module;

    std::string m_module_core_name;
};

// ********** Mdl_module_wait_queue  ***************************************************************
class Module_cache;

/// Used with module cache in order to allow parallel loading of modules.
class Mdl_module_wait_queue
{
    class Table;

public:
    /// For each module to load, an entry is created in the waiting table so threads that
    /// depend on that module can wait until the loading thread is finished
    class Entry
    {
    public:
        /// Constructor.
        ///
        /// \param name                 The name of the module to load.
        /// \param cache                The current instance of the module cache.
        /// \param parent_table         The table this entry belongs to.
        explicit Entry(
            std::string name,
            const Module_cache* cache,
            Table* parent_table);

        /// Destructor.
        ~Entry();

        /// Called when the module is currently loaded by another threads.
        /// The usage count must have already been incremented.
        /// Blocks until the loading thread calls \c notify.
        /// Decrements the usage count afterwards and, if this was the last usage of the entry,
        /// it self-destructs.
        ///
        /// \param cache               The current instance of the module cache.
        /// \return                    The result code the loading thread provided when call \c
        ///                            notify. The waiting thread needs to abort it's own loading
        ///                            process in case of an error.
        mi::Sint32 wait(const Module_cache* cache);

        /// Called by the loading thread after loading is done to wake the waiting threads.
        /// Decrements the usage count afterwards and, if this was the last usage of the entry,
        /// it self-destructs.
        ///
        /// \param cache               The current instance of the module cache.
        /// \param result_code         The result code that is passed to the waiting threads.
        void notify(
            const Module_cache* cache,
            mi::Sint32 result_code);

        /// Check if this module is loaded by the current thread.
        ///
        /// \param cache                The current instance of the module cache.
        /// \return                     True when the provided cache has the same context id.
        bool processed_in_current_context(const Module_cache* cache) const;

        /// Increments the usage counter of the entry.
        void increment_usage_count();

    private:
        /// Erases this entry from the parent table and self-destructs.
        void cleanup();

        std::string m_core_name;
        size_t m_cache_context_id;
        mi::base::Handle<mi::neuraylib::IMdl_loading_wait_handle> m_handle;
        Table* m_parent_table;
        std::mutex m_usage_count_mutex;
        size_t m_usage_count;
    };

    //---------------------------------------------------------------------------------------------
private:
    /// Table of waiting entries created for each transaction.
    class Table
    {
    public:
        /// Constructor.
        explicit Table() = default;

        /// Removes an entry from the table.
        ///
        /// \param name             The name of the entry (module to load).
        void erase(const std::string& name);

        /// Get or create a waiting entry for a module to load.
        /// Assumes that the table is already locked by the current thread.
        ///
        /// \param cache        The current module cache.
        /// \param name         The name of the module to load.
        /// \param out_created  Will be true after calling when the a new entry was created.
        /// \return             The entry to call wait on, in case it existed (outside the lock).
        Entry* get_waiting_entry(
            const Module_cache* cache,
            const std::string& name,
            bool& out_created);

        /// Get the number of entries in the table.
        mi::Size size() const;

        /// Check if this module is loaded by the current thread.
        bool processed_in_current_context(
            const Module_cache* cache,
            const std::string& name);

    private:
        std::mutex m_mutex;
        ankerl::unordered_dense::map<std::string, Entry*> m_elements;
    };


public:
    //---------------------------------------------------------------------------------------------

    /// Return structure for the \c lockup method
    struct Queue_lockup
    {
        /// If the module is already in the cache, \c nullptr otherwise
        mi::mdl::IModule const* cached_module;

        /// If the module is not in the cache, \c wait has to be called on this queue entry
        /// If this pointer is \c nullptr, too, the current thread is responsible for loading the
        /// module.
        Entry* queue_entry;
    };

    //---------------------------------------------------------------------------------------------

    /// Wraps the cache lookup_db and creates a waiting entry for a module to load.
    ///
    /// \param cache            The current module cache.
    /// \param transaction      The current transaction to use.
    /// \param name             The name of the module to load.
    /// \return                 The module, a waiting entry, or both \c nullptr which means the
    ///                         module has to be loaded on this thread.
    Queue_lockup lookup(
        const Module_cache* cache,
        size_t transaction,
        const std::string& name);

    /// Check if this module is loaded by the current thread.
    /// \param cache            The current module cache.
    /// \param transaction      The current transaction to use.
    /// \param name             The name of the module to load.
    ///
    /// \return                 True when the module is loaded in the current context.
    bool processed_in_current_context(
        const Module_cache* cache,
        size_t transaction,
        const std::string& name);

    /// Notify waiting threads about the finishing of the loading process of module.
    /// This function has to be called after a successfully loaded thread was registered in the
    /// Database (or a corresponding cache structure) AND also in case of loading failures.
    ///
    /// \param cache            The current module cache.
    /// \param transaction      The current transaction to use.
    /// \param module_name      The name of the module that has been processed.
    /// \param result_code      0 in case of success
    void notify(
        Module_cache* cache,
        size_t transaction,
        const std::string& module_name,
        int result_code);

    /// Try free this table when the transaction is not used anymore.
    /// \param transaction      The current transaction that specifies the table to cleanup.
    void cleanup_table(size_t transaction);

private:
    ankerl::unordered_dense::map<size_t, Table*> m_tables;
    std::mutex m_mutex;
};

// ********** Module_cache_lookup_handle ***********************************************************

class Module_cache_lookup_handle : public mi::mdl::IModule_cache_lookup_handle
{
public:
    explicit Module_cache_lookup_handle();
    virtual ~Module_cache_lookup_handle() = default;

    void set_lookup_name(const char* name);
    const char* get_lookup_name() const override;

    bool is_processing() const override;
    void set_is_processing(bool value);

private:
    std::string m_lookup_name;
    bool m_is_processing;
};

// ********** Module_cache *************************************************************************

/// Adapts the DB (or rather a transaction) to the IModule_cache interface.
class Module_cache : public mi::mdl::IModule_cache
{
private:
    static std::atomic<size_t> s_context_counter;

public:
    /// Default implementation of the IMdl_loading_wait_handle interface.
    /// Used by the \c Wait_handle_factory if no other factory is registered at the module cache.
    class Wait_handle
        : public mi::base::Interface_implement<mi::neuraylib::IMdl_loading_wait_handle>
    {
    public:
        explicit Wait_handle();

        void wait() override;
        void notify(mi::Sint32 result_code) override;
        mi::Sint32 get_result_code() const override;

    private:
        std::condition_variable m_conditional;
        std::mutex m_conditional_mutex;
        bool m_processed;
        int m_result_code;
    };

    /// Default implementation of the IMdl_loading_wait_handle_factory interface.
    /// Uses \c Wait_handles if no other factory is registered at the module cache.
    class Wait_handle_factory
        : public mi::base::Interface_implement<mi::neuraylib::IMdl_loading_wait_handle_factory>
    {
    public:
        explicit Wait_handle_factory() {}

        mi::neuraylib::IMdl_loading_wait_handle* create_wait_handle() const override;
    };

    explicit Module_cache(
        DB::Transaction* transaction,
        Mdl_module_wait_queue* queue,
        DB::Tag_set module_ignore_list);

    virtual ~Module_cache();

    /// Create an \c IModule_cache_lookup_handle for this \c IModule_cache implementation.
    mi::mdl::IModule_cache_lookup_handle* create_lookup_handle() const override;

    /// Free a handle created by \c create_lookup_handle.
    void free_lookup_handle(mi::mdl::IModule_cache_lookup_handle* handle) const override;

    /// If the DB contains the MDL module \p module_name, return it, otherwise \c nullptr.
    ///
    /// In case of access from multiple threads, only the first thread that wants to load module
    /// will return \c nullptr immediately, further threads will block until notify was called by
    /// the loading thread. In case loading failed on a different thread, \c lookup will also
    /// return \c nullptr after returning from waiting. The caller can check whether the current
    /// thread is supposed to load the module by calling \c processed_on_this_thread.
    ///
    /// \param module_name  The core name of the module to lookup.
    /// \param handle       A handle created by #create_lookup_handle() which is used throughout
    ///                     the loading process of a module or \c nullptr in case the goal is to
    ///                     just check if a module is loaded.
    const mi::mdl::IModule* lookup(
        const char* module_name,
        mi::mdl::IModule_cache_lookup_handle *handle) const override;

    /// Checks if the module is the DB. If so, the module is returned and \c nullptr otherwise.
    ///
    /// \param module_name  The core name of the module to lookup.
    const mi::mdl::IModule* lookup_db(const char* module_name) const;

    /// Checks if the module is the DB. If so, the module is returned and \c nullptr otherwise.
    const mi::mdl::IModule* lookup_db(DB::Tag tag) const;

    /// Check if this module is loading was started in the current context. I.e., on the thread
    /// that created this module cache instance.
    /// Note, this assumes a light weight implementation of the Cache. One instance for each
    /// call to \c load module.
    ///
    /// \param module_name      The core name of the module that is been processed.
    /// \return                 True if the module loading was initialized with this instance.
    bool loading_process_started_in_current_context(const char* module_name) const;

    /// Notify waiting threads about the finishing of the loading process of module.
    /// This function has to be called after a successfully loaded module was registered in the
    /// Database (or a corresponding cache structure) AND also in case of loading failures.
    ///
    /// \param module_name      The core name of the module that has been processed.
    /// \param result_code      Return code of the loading process. 0 in case of success.
    virtual void notify(const char* module_name, int result_code);

    /// Get the module loading callback
    mi::mdl::IModule_loaded_callback* get_module_loading_callback() const override;

    /// Set the module loading callback.
    void set_module_loading_callback(mi::mdl::IModule_loaded_callback* callback);

    /// Get the module cache wait handle factory.
    const mi::neuraylib::IMdl_loading_wait_handle_factory* get_wait_handle_factory() const;

    /// Set the module cache wait handle factory.
    void set_wait_handle_factory(const mi::neuraylib::IMdl_loading_wait_handle_factory* factory);

    /// Get an unique identifier for the context in which the current module is loaded.
    ///
    /// \return                 The identifier.
    size_t get_loading_context_id() const { return m_context_id; }

private:

    size_t m_context_id;
    DB::Transaction* m_transaction;
    Mdl_module_wait_queue* m_queue;
    mi::mdl::IModule_loaded_callback* m_module_load_callback;
    mi::base::Handle<const mi::neuraylib::IMdl_loading_wait_handle_factory>
        m_default_wait_handle_factory;
    mi::base::Handle<const mi::neuraylib::IMdl_loading_wait_handle_factory>
        m_user_wait_handle_factory;
    DB::Tag_set m_ignore_list;
};

// ********** Call_evaluator ***********************************************************************

/// Evaluates calls during material compilation.
///
/// Used to fold resource-related calls into constants.
///
/// \tparam T   Either mi::mdl::IGenerated_code_dag or mi::mdl::ILambda_function.
template<typename T>
class Call_evaluator : public mi::mdl::ICall_evaluator
{
public:
    /// Constructor.
    ///
    /// \param owner                    the owner of a resource
    /// \param transaction              the current transaction
    /// \param has_resource_attributes  true, if resource attributes can be folded
    Call_evaluator(
        const T* owner,
        DB::Transaction* transaction,
        bool has_resource_attributes)
      : m_owner( owner),
        m_transaction( transaction),
        m_has_resource_attributes( has_resource_attributes)
    {}

    /// Destructor.
    virtual ~Call_evaluator() {}

    /// Checks whether evaluate_intrinsic_function() should be called for an unhandled
    /// intrinsic function with the given semantic.
    ///
    /// \param semantic  the semantic to check for
    bool is_evaluate_intrinsic_function_enabled(
        mi::mdl::IDefinition::Semantics semantic) const final;

    /// Called by IExpression_call::fold() to evaluate unhandled intrinsic functions.
    ///
    /// \param value_factory  The value factory used to create the result value
    /// \param semantic       The semantic of the function to call
    /// \param arguments      The arguments for the call
    /// \param n_arguments    The number of arguments
    ///
    /// \return               IValue_bad if this function could not be evaluated, otherwise
    ///                       its folded value.
    const mi::mdl::IValue* evaluate_intrinsic_function(
        mi::mdl::IValue_factory* value_factory,
        mi::mdl::IDefinition::Semantics semantic,
        const mi::mdl::IValue* const arguments[],
        size_t n_arguments) const final;

private:
    /// Folds df::light_profile_power() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_df_light_profile_power(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds df::light_profile_maximum() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_df_light_profile_maximum(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds df::light_profile_isinvalid() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_df_light_profile_isvalid(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds df::bsdf_measurement_isvalid() to a constant, or returns IValue_bad in case of
    /// errors.
    const mi::mdl::IValue* fold_df_bsdf_measurement_isvalid(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds tex::width() to a constant, or returns IValue_bad in case of errors.
    ///
    /// \param uvtile_arg   May be \c nullptr for non-uvtile texture calls.
    /// \param frame_arg    May be \c nullptr for non-animated texture calls.
    const mi::mdl::IValue* fold_tex_width(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument,
        const mi::mdl::IValue* uvtile_arg,
        const mi::mdl::IValue* frame_arg) const;

    /// Folds tex::height() to a constant, or returns IValue_bad in case of errors.
    ///
    /// \param uvtile_arg   May be \c nullptr for non-uvtile texture calls.
    /// \param frame_arg    May be \c nullptr for non-animated texture calls.
    const mi::mdl::IValue* fold_tex_height(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument,
        const mi::mdl::IValue* uvtile_arg,
        const mi::mdl::IValue* frame_arg) const;

    /// Folds tex::depth() to a constant, or returns IValue_bad in case of errors.
    ///
    /// \p frame_arg may be \c nullptr for non-animated texture calls.
    const mi::mdl::IValue* fold_tex_depth(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument,
        const mi::mdl::IValue* frame_arg) const;

    /// Folds tex::texture_invalid() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_tex_texture_isvalid(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds tex::width_offset() to a constant, or returns IValue_bad in case of errors.
    ///
    /// The current implementation always returns an IValue_int with 0.
    const mi::mdl::IValue* fold_tex_width_offset(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* tex,
        const mi::mdl::IValue* offset) const;

    /// Folds tex::height_offset() to a constant, or returns IValue_bad in case of errors.
    ///
    /// The current implementation always returns an IValue_int with 0.
    const mi::mdl::IValue* fold_tex_height_offset(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* tex,
        const mi::mdl::IValue* offset) const;

    /// Folds tex::depth_offset() to a constant, or returns IValue_bad in case of errors.
    ///
    /// The current implementation always returns an IValue_int with 0.
    const mi::mdl::IValue* fold_tex_depth_offset(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* tex,
        const mi::mdl::IValue* offset) const;

    /// Folds tex::first_frame() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_tex_first_frame(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds tex::last_frame() to a constant, or returns IValue_bad in case of errors.
    const mi::mdl::IValue* fold_tex_last_frame(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* argument) const;

    /// Folds tex::grid_to_object_space() to a constant, or returns IValue_bad in case of errors.
    ///
    /// The current implementation always returns a 4x4 all-zero float matrix.
        const mi::mdl::IValue* fold_tex_grid_to_object_space(
        mi::mdl::IValue_factory* value_factory,
        const mi::mdl::IValue* tex,
        const mi::mdl::IValue* offset) const;

    const T* m_owner;
    DB::Transaction* m_transaction;
    bool m_has_resource_attributes;
};

// ********** Resource names ***********************************************************************

namespace DETAIL {

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
///
/// \param module_name    The MDL name of the owning module.
std::string unresolve_resource_filename(
    const char* filename,
    const char* module_filename,
    const char* module_name,
    MDL::Execution_context* context);

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
///
/// \param module_name    The MDL name of the owning module.
std::string unresolve_resource_filename(
    const char* archive_filename,
    const char* archive_membername,
    const char* module_filename,
    const char* module_name,
    MDL::Execution_context* context);

} // namespace DETAIL

/// Converts a hash from the core representation to the API representation.
mi::base::Uuid convert_hash( const unsigned char hash[16]);

/// Converts a hash from the API representation to the core representation.
bool convert_hash( const mi::base::Uuid& hash_in, unsigned char hash_out[16]);

/// Replaces the frame and/or uv-tile marker by coordinates of a given uv-tile.
///
/// \param s       String containing a valid frame and/or uv-tile marker.
/// \param f       The frame number of the uv-tile.
/// \param u       The u coordinate of the uv-tile.
/// \param v       The v coordinate of the uv-tile.
/// \return        String with the frame and/or uv-tile marker replaced by the coordinates of the
///                uv-tile, or the empty string in case of errors.
std::string frame_uvtile_marker_to_string( std::string s, mi::Size f, mi::Sint32 u, mi::Sint32 v);

/// Returns an absolute MDL file path for the given filename.
///
/// Does not check for existence. Returns the empty string on failure.
std::string get_file_path(
    std::string filename, mi::neuraylib::IMdl_impexp_api::Search_option option);

/// Converts the mi::mdl enum for gamma mode into a float (using 0.0f for gamma_default).
mi::Float32 convert_gamma_enum_to_float( mi::mdl::IValue_texture::gamma_mode gamma);

/// Converts a gamma value as float to the mi::mdl enum (using gamma_default for all values
/// different from 1.0f and 2.2f).
mi::mdl::IValue_texture::gamma_mode convert_gamma_float_to_enum( mi::Float32 gamma);

/// Functions in the core use the special value 0xffffffffU, although it is not part of the
/// corresponding enum.
const mi::mdl::IMDL::MDL_version mi_mdl_IMDL_MDL_VERSION_INVALID
    = static_cast<mi::mdl::IMDL::MDL_version>( 0xffffffffU);

/// Converts mi::mdl::IMDL::MDL_version into mi::neuraylib::Mdl_version.
mi::neuraylib::Mdl_version convert_mdl_version( mi::mdl::IMDL::MDL_version version);

/// Converts values from mi::mdl::IMDL::MDL_version plus
/// mi_mdl_IMDL_MDL_VERSION_INVALID into mi::neuraylib::Mdl_version.
mi::neuraylib::Mdl_version convert_mdl_version_uint32( mi::Uint32 version);

/// Converts mi::neuraylib::Mdl_version into mi::mdl::IMDL::MDL_version.
mi::mdl::IMDL::MDL_version convert_mdl_version( mi::neuraylib::Mdl_version version);

/// Returns a string representation of mi::mdl::IMDL::MDL_version.
const char* stringify_mdl_version( mi::mdl::IMDL::MDL_version version);

/// Splits the mi::mdl::IMDL::MDL_version into major and minor version parts.
std::pair<int,int> split_mdl_version( mi::mdl::IMDL::MDL_version version);

/// Combines major and minor version parts into mi::mdl::IMDL::MDL_version.
mi::mdl::IMDL::MDL_version combine_mdl_version( int major, int minor);

/// Creates a value referencing a texture identified by an MDL file path.
///
/// \param transaction   The transaction to be used.
/// \param file_path     The absolute MDL file path that identifies the texture. The MDL
///                      search paths are used to resolve the file path. See section 2.2 in
///                      [\ref MDLLS] for details.
/// \param shape         The value that is returned by #IType_texture::get_shape() on the type
///                      corresponding to the return value.
/// \param gamma         The value that is returned by #TEXTURE::Texture::get_gamma()
///                      on the DB element referenced by the return value.
/// \param selector      The selector (or \c nullptr).
/// \param shared        Indicates whether you want to re-use the DB elements for that texture
///                      if it has already been loaded, or if you want to create new DB elements
///                      in all cases. Note that sharing is based on the location where the
///                      texture is finally located and includes sharing with instances that
///                      have not explicitly been loaded via this method, e.g., textures in
///                      defaults.
/// \param context       Execution context. The error codes have the following meaning:
///                      -  0: Success.
///                      - -1: Invalid parameters (\c nullptr pointer).
///                      - -2: The file path is not an absolute MDL file path.
///                      - -3: Failed to resolve the given file path, or no suitable image
///                            plugin available.
/// \return              The value referencing the texture, or \c nullptr in case of failure.
IValue_texture* create_texture(
    DB::Transaction* transaction,
    const char* file_path,
    IType_texture::Shape shape,
    mi::Float32 gamma,
    const char* selector,
    bool shared,
    Execution_context* context);

/// Creates a value referencing a light profile identified by an MDL file path.
///
/// \param transaction   The transaction to be used.
/// \param file_path     The absolute MDL file path that identifies the light profile. The MDL
///                      search paths are used to resolve the file path. See section 2.2 in
///                      [\ref MDLLS] for details.
/// \param shared        Indicates whether you want to re-use the DB element for that light
///                      profile if it has already been loaded, or if you want to create a new
///                      DB element in all cases. Note that sharing is based on the location
///                      where the light profile is finally located and includes sharing with
///                      instances that have not explicitly been loaded via this method, e.g.,
///                      light profiles in defaults.
/// \param context       Execution context. The error codes have the following meaning:
///                      -  0: Success.
///                      - -1: Invalid parameters (\c nullptr pointer).
///                      - -2: The file path is not an absolute MDL file path.
///                      - -3: Failed to resolve the given file path.
/// \return              The value referencing the light profile, or \c nullptr in case of failure.
IValue_light_profile* create_light_profile(
    DB::Transaction* transaction,
    const char* file_path,
    bool shared,
    Execution_context* context);

/// Creates a value referencing a BSDF measurement identified by an MDL file path.
///
/// \param transaction   The transaction to be used.
/// \param file_path     The absolute MDL file path that identifies the BSDF measurement. The
///                      MDL search paths are used to resolve the file path. See section 2.2 in
///                      [\ref MDLLS] for details.
/// \param shared        Indicates whether you want to re-use the DB element for that BSDF
///                      measurement if it has already been loaded, or if you want to create a
///                      new DB element in all cases. Note that sharing is based on the location
///                      where the BSDF measurement is finally located and includes sharing with
///                      instances that have not explicitly been loaded via this method, e.g.,
///                      BSDF measurements in defaults.
/// \param context       Execution context. The error codes have the following meaning:
///                      -  0: Success.
///                      - -1: Invalid parameters (\c nullptr pointer).
///                      - -2: The file path is not an absolute MDL file path.
///                      - -3: Failed to resolve the given file path.
/// \return              The value referencing the BSDF measurement, or \c nullptr in case of
///                      failure.
IValue_bsdf_measurement* create_bsdf_measurement(
    DB::Transaction* transaction,
    const char* file_path,
    bool shared,
    Execution_context* context);

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_UTILITIES_H
