/***************************************************************************************************
 * Copyright (c) 2012-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <map>
#include <vector>
#include <set>
#include <thread>
#include <unordered_map>
#include <mutex>

#include <boost/any.hpp>

#include <mi/base/ilogger.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/neuraylib/typedefs.h>
#include <mi/neuraylib/ifunction_definition.h>
#include <mi/neuraylib/imdl_loading_wait_handle.h>
#include <mi/neuraylib/imdl_impexp_api.h>

#include <base/lib/log/i_log_assert.h>
#include <base/data/db/i_db_tag.h>

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
    class IMDL_resource_reader;
    class IMdle_input_stream;
    class IModule;
    class IModule_cache;
    class IModule_cache_wait_handle;
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
class IType;
class IType_list;
class IValue;
class IValue_list;
class Mdl_compiled_material;
class Mdl_function_definition;
class Mdl_module;

// **********  Computation of references to other DB element ***************************************

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


// **********  Memory allocation helper class ******************************************************

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

// **********  Name parsing/splitting **************************************************************

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

// **********  Misc utility functions **************************************************************

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

/// Returns the DB name used for the builtins module ("mdl::<builtins>" or "mdl::%3Cbuiltins%3E").
const char* get_builtins_module_db_name();

/// Returns the MDL name used for the builtins module ("::<builtins>" or "::%3Cbuiltins%3E").
const char* get_builtins_module_mdl_name();

/// Returns the simple name used for the builtins module ("<builtins>" or "%3Cbuiltins%3E").
const char* get_builtins_module_simple_name();

/// Returns the DB name used for the neuray module ("mdl::<neuray>" or "mdl::%3Cneuray%3E").
const char* get_neuray_module_db_name();

/// Returns the MDL name used for the neuray module ("::<neuray>" or "::%3Cneuray%3E").
const char* get_neuray_module_mdl_name();

/// Indicates whether encoded names are enabled. This flag also enables material names with
/// signatures and removes the extra slash for MDLE names on Windows.
bool get_encoded_names_enabled();

/// Enabled or disabled encoded names.
void set_encoded_names_enabled( bool value);

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
/// Tries to find the name in the code DAG (if not NULL), and uses the DB as fallback.
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
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c NULL (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The serialized function definition (and module) name, or \c NULL in
///                          case of errors. The method always returns \c NULL if encoded
///                          names are disabled.
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
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c NULL (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized function name and argument types, or \c NULL in case
///                          of errors. The method always returns \c NULL if encoded names are
///                          disabled.
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
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c NULL (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized function name and argument types, or \c NULL in case
///                          of errors. The method always returns \c NULL if encoded names are
///                          disabled.
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
/// \param mdle_callback     A callback to map the filename of MDLE modules. Ignored for
///                          non-MDLE modules. Can be \c NULL (which is treated like a callback
///                          implementing the identity transformation).
/// \return                  The deserialized module name, or \c NULL in case of errors. The method
///                          always returns \c NULL if encoded names are disabled.
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


/// Indicates whether a function of the given semantic is a valid prototype for prototype-based
/// functions or variants.
bool is_supported_prototype(
    mi::neuraylib::IFunction_definition::Semantics sema, bool for_variant);

/// Returns a default compiled material.
///
/// Can be used as a default for failed compilation steps. Returns \c NULL if someone explicitly
/// removed the corresponding material definition from the DB without removing the entire module
/// (should not happen).
Mdl_compiled_material* get_default_compiled_material( DB::Transaction* transaction);

/// Represents an MDL compiler message. Similar to mi::mdl::IMessage.
class Message
{
public:

    enum Kind{
        MSG_COMILER_CORE,
        MSG_COMILER_BACKEND,
        MSG_COMPILER_DAG,
        MSG_COMPILER_ARCHIVE_TOOL,
        MSG_IMP_EXP,
        MSG_INTEGRATION,
        MSG_UNCATEGORIZED,
        MSG_FORCE_32_BIT = 0xffffffffU
    };

    Message( mi::base::Message_severity severity, const std::string& message)
        : m_severity( severity)
        , m_code(-1)
        , m_message( message)
        , m_kind(MSG_UNCATEGORIZED) { }

    Message(mi::base::Message_severity severity, const std::string& message, mi::Sint32 code, Kind kind)
        : m_severity(severity)
        , m_code(code)
        , m_message(message)
        , m_kind(kind) { }

    Message(const mi::mdl::IMessage *message);

    mi::base::Message_severity  m_severity;
    mi::Sint32                  m_code;
    std::string                 m_message;
    Kind                        m_kind;
    std::vector<Message>        m_notes;

};

/// Simple Option class.
class Option
{
public:

    using Validator = bool (*)(const boost::any&);

    Option(const std::string& name, const boost::any& default_value, bool is_interface, Validator validator=nullptr)
        : m_name(name)
        , m_value(default_value)
        , m_default_value(default_value)
        , m_validator(validator)
        , m_is_interface(is_interface)
        , m_is_set(false)
    {}

    const char* get_name() const
    {
        return m_name.c_str();
    }

    bool set_value(const boost::any& value)
    {
        if(m_validator && !m_validator(value)) {
            return false;
        }
        m_value = value;
        m_is_set = true;
        return true;
    }

    const boost::any& get_value() const
    {
        return m_value;
    }

    void reset()
    {
        m_value = m_default_value;
        m_is_set = false;
    }

    bool is_set() const
    {
        return m_is_set;
    }

    bool is_interface() const
    {
        return m_is_interface;
    }

private:

    std::string m_name;
    boost::any m_value;
    boost::any m_default_value;
    Validator m_validator;
    bool m_is_interface;
    bool m_is_set;
};

// When adding new options
// - adapt the Execution_context constructor to register the default and its type
// - adapt create_thread_context() if necessary

#define MDL_CTX_OPTION_WARNING                            "warning"
#define MDL_CTX_OPTION_OPTIMIZATION_LEVEL                 "optimization_level"
#define MDL_CTX_OPTION_INTERNAL_SPACE                     "internal_space"
#define MDL_CTX_OPTION_FOLD_METERS_PER_SCENE_UNIT         "fold_meters_per_scene_unit"
#define MDL_CTX_OPTION_METERS_PER_SCENE_UNIT              "meters_per_scene_unit"
#define MDL_CTX_OPTION_WAVELENGTH_MIN                     "wavelength_min"
#define MDL_CTX_OPTION_WAVELENGTH_MAX                     "wavelength_max"
#define MDL_CTX_OPTION_INCLUDE_GEO_NORMAL                 "include_geometry_normal"
#define MDL_CTX_OPTION_BUNDLE_RESOURCES                   "bundle_resources"
#define MDL_CTX_OPTION_EXPERIMENTAL                       "experimental"
#define MDL_CTX_OPTION_RESOLVE_RESOURCES                  "resolve_resources"
#define MDL_CTX_OPTION_FOLD_TERNARY_ON_DF                 "fold_ternary_on_df"
#define MDL_CTX_OPTION_IGNORE_NOINLINE                    "ignore_noinline"
#define MDL_CTX_OPTION_REMOVE_DEAD_PARAMETERS             "remove_dead_parameters"
#define MDL_CTX_OPTION_FOLD_ALL_BOOL_PARAMETERS           "fold_all_bool_parameters"
#define MDL_CTX_OPTION_FOLD_ALL_ENUM_PARAMETERS           "fold_all_enum_parameters"
#define MDL_CTX_OPTION_FOLD_PARAMETERS                    "fold_parameters"
#define MDL_CTX_OPTION_FOLD_TRIVIAL_CUTOUT_OPACITY        "fold_trivial_cutout_opacity"
#define MDL_CTX_OPTION_FOLD_TRANSPARENT_LAYERS            "fold_transparent_layers"
#define MDL_CTX_OPTION_SERIALIZE_CLASS_INSTANCE_DATA      "serialize_class_instance_data"
#define MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY        "loading_wait_handle_factory"
#define MDL_CTX_OPTION_DEPRECATED_REPLACE_EXISTING        "replace_existing"
#define MDL_CTX_OPTION_TARGET_MATERIAL_MODEL_MODE         "target_material_model_mode"
// Not documented in the API (used by the module transformer, but not for general use).
#define MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS  "keep_original_resource_file_paths"

/// Represents an MDL execution context. Similar to mi::mdl::Thread_context.
class Execution_context
{
public:

    Execution_context();

    mi::Size get_messages_count() const;

    mi::Size get_error_messages_count() const;

    const Message& get_message(mi::Size index) const;

    const Message& get_error_message(mi::Size index) const;

    void add_message(const mi::mdl::IMessage* message);

    void add_error_message(const mi::mdl::IMessage* message);

    void add_message(const Message& message);

    void add_error_message(const Message& message);

    void add_messages(const mi::mdl::Messages& messages);

    void clear_messages();

    mi::Size get_option_count() const;

    mi::Size get_option_index(const std::string& name) const;

    const char* get_option_name(mi::Size index) const;

    // The template parameter T has to match the option's value type, otherwise an exception might
    // be thrown.
    template<typename T>
    T get_option(const std::string& name) const {

        mi::Size index = get_option_index(name);
        ASSERT(M_SCENE, index < m_options.size());

        const Option& option = m_options[index];
        ASSERT(M_SCENE, !option.is_interface());

        return boost::any_cast<T> (option.get_value());
    }

    template<typename T>
    T* get_interface_option(const std::string& name) const
    {
        mi::Size index = get_option_index(name);
        ASSERT(M_SCENE, index < m_options.size());

        const Option& option = m_options[index];
        ASSERT(M_SCENE, option.is_interface());

        mi::base::Handle<const mi::base::IInterface> handle
            = boost::any_cast<mi::base::Handle<const mi::base::IInterface>>(option.get_value());

        if (!handle)
            return nullptr;

        mi::base::Handle<T> value(handle.get_interface<T>());
        if (!value)
            return nullptr;

        value->retain();
        return value.get();
    }

    mi::Sint32 get_option(const std::string& name, boost::any& value) const;

    mi::Sint32 set_option(const std::string& name, const boost::any& value);

    void set_result(mi::Sint32 result);

    mi::Sint32 get_result() const;

private:

    void add_option(const Option& option);

    std::vector<Message> m_messages;
    std::vector<Message> m_error_messages;

    std::map<std::string, mi::Size> m_options_2_index;
    std::vector<Option> m_options;

    mi::Sint32 m_result;

};

/// Adds MDL messages to an execution context.
void convert_messages( const mi::mdl::Messages& in_messages, Execution_context* context);

/// Logs the messages in an execution context.
void log_messages( const Execution_context* context);

/// Adds MDL messages to an optional execution context and logs them.
///
/// Similar to #convert_messages() followed by #log_messages(), except that context can be \c NULL.
void convert_and_log_messages( const mi::mdl::Messages& in_messages, Execution_context* context);

/// Adds messages from an execution context to the mi::mdl::Messages.
void convert_messages( const Execution_context* context, mi::mdl::Messages& out_messages);

/// Adds \p message as message to the context.
///
/// If the severity is #mi::base::MESSAGE_SEVERITY_ERROR or #mi::base::MESSAGE_SEVERITY_FATAL, then
/// the message is also added as error message to the context, and the context result is set to
/// \p result.
///
/// Does nothing if \p context is \c NULL. Returns \p result.
mi::Sint32 add_message( Execution_context* context, const Message& message, mi::Sint32 result);

/// Adds \p message as message and error message to the context, and sets the result to \p result.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and -1 as message code.
/// Does nothing if \p context is \c NULL. Returns \p result.
mi::Sint32 add_error_message(
    Execution_context* context, const std::string& message, mi::Sint32 result);

/// Adds \p message as warning message to the context.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and -1 as message code.
/// Does nothing if \p context is \c NULL. Returns \p result.
void add_warning_message( Execution_context* context, const std::string& message);

/// Adds \p message as info message to the context.
///
/// Uses #Message::MSG_INTEGRATION as message kind, and -1 as message code.
/// Does nothing if \p context is \c NULL. Returns \p result.
void add_info_message( Execution_context* context, const std::string& message);

/// Wraps an MDL input stream as IReader.
mi::neuraylib::IReader* get_reader( mi::mdl::IInput_stream* stream);

/// Wraps an MDL resource reader as IReader.
mi::neuraylib::IReader* get_reader( mi::mdl::IMDL_resource_reader* resource_reader);

/// Wraps an IReader as MDL input stream.
mi::mdl::IInput_stream* get_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename);

/// Wraps an IReader as MDL MDLE input stream.
mi::mdl::IMdle_input_stream* get_mdle_input_stream(
    mi::neuraylib::IReader* reader, const std::string& filename);

/// Wraps an IReader as MDL resource reader. The reader needs to support absolute access.
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


// **********  Resource-related attributes *********************************************************

/// Retrieve the attributes of a texture resource.
///
/// \param transaction      The DB transaction to use.
/// \param tex_tag          A texture tag.
/// \param uvtile_x         The x-coordinate of the uvtile (or 0 for non-uvtiles).
/// \param uvtile_y         The y-coordinate of the uvtile (or 0 for non-uvtiles).
/// \param[out] valid       The result of texture_isvalid().
/// \param[out] width       The width of \p texture.
/// \param[out] height      The height of \p texture.
/// \param[out] depth       The depth of \p texture.
void get_texture_attributes(
    DB::Transaction* transaction,
    DB::Tag tex_tag,
    mi::Sint32 uvtile_x,
    mi::Sint32 uvtile_y,
    bool& valid,
    int& width,
    int& height,
    int& depth);

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


// **********  Mdl_dag_builder *********************************************************************

/// Helper class to handle conversion from MDL::IExpression into mi::mdl::DAG_node.
///
/// \tparam T   Either mi::mdl::IDag_builder or mi::mdl::IGenerated_code_dag::DAG_node_factory.
///
/// \note The class behaves differently for temporaries. If T == IDag_builder, temporaries are
///       re-converted each time they are encountered. If T == IGenerated_code_dag::DAG_node_factory,
///       temporaries are converted only once, and referenced via DAG_temporary nodes.
template<class T>
class Mdl_dag_builder
{
public:
    /// Constructor.
    ///
    /// \param transaction                 The DB transaction to use (needed to access attached
    ///                                    function calls or material instances).
    /// \param dag_builder                 A DAG builder used to construct the DAG nodes.
    /// \param compiled_material           The compiled material that will be used to resolve
    ///                                    temporaries. Can be \c NULL if the expressions to be
    ///                                    converted do not contain any references to temporaries.
    Mdl_dag_builder(
        DB::Transaction* transaction,
        T* dag_builder,
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
    /// \param mdl_type                    The MDL type corresponding to \p expr
    /// \param expr                        The expression to convert.
    /// \return                            The created MDL DAG node, or \c NULL in case of failures,
    ///                                    e.g., argument type mismatches (including further down
    ///                                    the call graph).
    const mi::mdl::DAG_node* int_expr_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression* expr);

    /// Returns the cached converted temporaries.
    const std::vector<const mi::mdl::DAG_node*>& get_temporaries() const
    { return m_temporaries; }

    /// Returns the cached parameter types.
    const std::vector<const mi::mdl::IType*>& get_parameter_types() const
    { return m_parameter_types; }

private:
    const mi::mdl::DAG_node* int_expr_constant_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression_constant* expr);

    const mi::mdl::DAG_node* int_expr_parameter_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression_parameter* expr);

    const mi::mdl::DAG_node* int_expr_call_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression_call* expr);

    const mi::mdl::DAG_node* int_expr_direct_call_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression_direct_call* expr);

    const mi::mdl::DAG_node* int_expr_temporary_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const IExpression_temporary* expr);

    /// Clones a DAG node.
    ///
    /// \param node  The DAG IR node to clone.
    /// \return      The clone of \p node.
    const mi::mdl::DAG_node* clone_dag_node( const mi::mdl::DAG_node* node);

private:
    /// Shared between #int_expr_call_to_mdl_dag_node() and
    /// #int_expr_direct_call_to_mdl_dag_node().
    ///
    /// \param call_name    DB name of the corresponding function definition.
    const mi::mdl::DAG_node* int_expr_call_to_mdl_dag_node_shared(
        const mi::mdl::IType* mdl_type,
        const Mdl_module* module,
        bool is_material,
        mi::Size definition_index,
        const char* call_name,
        const IExpression_list* arguments);

    /// Adds \p value to m_converted_call_expressions and returns it.
    const mi::mdl::DAG_node* add_cache_entry( DB::Tag tag, const mi::mdl::DAG_node* value);

    /// The DB transaction to use (needed to access attached function calls or material instances).
    DB::Transaction* m_transaction;
    /// A DAG builder used to construct the DAG nodes.
    T* m_dag_builder;
    /// The type factory of the DAG builder.
    mi::mdl::IType_factory* m_type_factory;
    /// The value factory of the DAG builder.
    mi::mdl::IValue_factory* m_value_factory;
    /// If non-NULL, a compiled material that will be used to resolve temporaries.
    const Mdl_compiled_material* m_compiled_material;
    /// The converted temporaries.
    std::vector<const mi::mdl::DAG_node*> m_temporaries;
    /// The MDL types of the parameter references.
    std::vector<const mi::mdl::IType*> m_parameter_types;
    /// Set of indirect calls in the current call stack, used to check for cycles.
    std::set<DB::Tag> m_set_indirect_calls;
    /// Cache of already converted function calls or material instances.
    std::map<DB::Tag, const mi::mdl::DAG_node*> m_converted_call_expressions;
};


// **********  Mdl_call_resolver *******************************************************************

/// Finds the owning module of a function definition.
class Mdl_call_resolver : public mi::mdl::ICall_name_resolver
{
public:
    /// Constructor.
    ///
    /// \param transaction  the transaction to use for name lookup
    Mdl_call_resolver(DB::Transaction* transaction) : m_transaction( transaction) { }

    virtual ~Mdl_call_resolver();

    /// Finds the owning module of a function definition.
    ///
    /// \param name   The core name of a function definition.
    /// \return       The owning module, or \c NULL in case of failures.
    const mi::mdl::IModule* get_owner_module(const char* name) const override;

    /// Find the owner code DAG of a given entity name.
    /// If the entity name does not contain a colon, you should return the builtins DAG,
    /// which you can identify by calling its owner module's IModule::is_builtins().
    ///
    /// \param entity_name    the entity name
    ///
    /// \returns the owning module of this entity if found, NULL otherwise
    const mi::mdl::IGenerated_code_dag* get_owner_dag(const char* entity_name) const override;

private:
    DB::Tag get_module_tag(const char* entity_name) const;

private:
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
    /// \return       The owning module, or \c NULL in case of failures.
    const mi::mdl::IModule* get_owner_module(const char* name) const override;

private:
    const mi::mdl::IModule* m_module;

    std::string m_module_core_name;
};

// **********  Mdl_module_wait_queue  **************************************************************
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
            const std::string& name,
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
        explicit Table();

        /// Destructor.
        ~Table() = default;

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
        std::unordered_map<std::string, Entry*> m_elements;
    };


public:
    //---------------------------------------------------------------------------------------------

    /// Return structure for the \c lockup method
    struct Queue_lockup
    {
        /// If the module is already in the cache, NULL otherwise
        mi::mdl::IModule const* cached_module;

        /// If the module is not in the cache, \c wait has to be called on this queue entry
        /// If this pointer is NULL, too, the current thread is responsible for loading the module.
        Entry* queue_entry;
    };

    //---------------------------------------------------------------------------------------------

    /// Wraps the cache lookup_db and creates a waiting entry for a module to load.
    ///
    /// \param cache            The current module cache.
    /// \param transaction      The current transaction to use.
    /// \param name             The name of the module to load.
    /// \return                 The module, a waiting entry, or both NULL which means the module
    ///                         has to be loaded on this thread.
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
    std::unordered_map<size_t, Table*> m_tables;
    std::mutex m_mutex;
};

/// Adds an error message to the given execution context
/// \param context  the execution context.
/// \param message  the message string.
/// \param result   an error code which will be set as the current context result.
mi::Sint32 add_context_error(
    MDL::Execution_context* context,
    const std::string& message,
    mi::Sint32 result);

// **********  Resource names **********************************************************************

namespace DETAIL {

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
///
/// \param module_name    The MDL name of the owning module.
std::string unresolve_resource_filename(
    const char* filename, const char* module_filename, const char* module_name);

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
///
/// \param module_name    The MDL name of the owning module.
std::string unresolve_resource_filename(
    const char* archive_filename,
    const char* archive_membername,
    const char* module_filename,
    const char* module_name);

} // namespace DETAIL

/// Converts a hash from the MDL API representation to the base API representation.
mi::base::Uuid convert_hash( const unsigned char hash[16]);

/// Converts a hash from the base API representation to the MDL API representation.
bool convert_hash( const mi::base::Uuid& hash_in, unsigned char hash_out[16]);

/// Replaces the uv-tile marker by coordinates of a given uv-tile.
///
/// \param s       String containing a valid uv-tile marker.
/// \param u       The u coordinate of the uv-tile.
/// \param u       The v coordinate of the uv-tile.
/// \return        String with the uv-tile marker replaced by the coordinates of the uv-tile, or
///                the empty string in case of errors.
std::string uvtile_marker_to_string( const std::string& s, mi::Sint32 u, mi::Sint32 v);

/// Replaces the pattern describing the coordinates of a uv-tile by the given marker.
///
/// \param s        String containing the index pattern, e.g., "_u1_v1"
/// \param marker   The marker to replace the pattern with, e.g., "<UVTILE1>"
/// \return         The string with the index pattern replaced by the marked (if found), or the
///                 empty string in case of errors.
std::string uvtile_string_to_marker( const std::string& s, const std::string& marker);

/// Returns an absolute MDL file path for the given filename.
///
/// Does not check for existence. Returns the empty string on failure.
std::string get_file_path(
    std::string filename, mi::neuraylib::IMdl_impexp_api::Search_option option);

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_UTILITIES_H

