/***************************************************************************************************
 * Copyright (c) 2012-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/base/ilogger.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/neuraylib/typedefs.h>

#include <base/lib/log/i_log_assert.h>
#include <base/data/db/i_db_tag.h>
#include <base/system/stlext/i_stlext_any.h>

namespace mi { namespace neuraylib { class IReader; } }
namespace mi { namespace mdl { class IMDL_resource_reader; } }

namespace MI {

namespace DB { class Transaction; }
namespace IMAGE { class IMdr_callback; }

namespace MDL {

class IValue;
class IValue_list;
class IExpression;
class IExpression_call;
class IExpression_constant;
class IExpression_direct_call;
class IExpression_factory;
class IExpression_list;
class IExpression_parameter;
class IExpression_temporary;
class IAnnotation;
class IAnnotation_block;
class IAnnotation_list;
class Mdl_compiled_material;
class Mdl_function_definition;
class Mdl_material_definition;


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
        : m_data(NULL)
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

// **********  Misc utility functions **************************************************************

/// Returns the DB element name used for the array constructor.
const char* get_array_constructor_db_name();

/// Returns the MDL name used for the array constructor.
const char* get_array_constructor_mdl_name();

/// Returns \c true for builtin modules.
///
/// Builtin modules have no underlying file, e.g., "::state".
///
/// \param module   The name of the module (including leading double colons, e.g., "::state").
bool is_builtin_module( const std::string& module);


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

    typedef bool(*Validator)(const STLEXT::Any& v);

    Option(const std::string& name, const STLEXT::Any& default_value, Validator validator=nullptr)
        : m_name(name)
        , m_value(default_value)
        , m_default_value(default_value)
        , m_validator(validator)
        , m_is_set(false)
    {}

    const char* get_name() const
    {
        return m_name.c_str();
    }

    bool set_value(const STLEXT::Any& value) 
    {
        if(m_validator && !m_validator(value)) {
            return false;
        }
        m_value = value;
        m_is_set = true;
        return true;
    }

    const STLEXT::Any& get_value() const
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

private:

    std::string m_name;
    STLEXT::Any m_value;
    STLEXT::Any m_default_value;
    Validator m_validator;
    bool m_is_set;
};

/// Represents an MDL execution context. Similar to mi::mdl::Thread_context.
class Execution_context
{
public:

    #define MDL_CTX_OPTION_INTERNAL_SPACE           "internal_space"
    #define MDL_CTX_OPTION_METERS_PER_SCENE_UNIT    "meters_per_scene_unit"
    #define MDL_CTX_OPTION_WAVELENGTH_MIN           "wavelength_min"
    #define MDL_CTX_OPTION_WAVELENGTH_MAX           "wavelength_max"
    #define MDL_CTX_OPTION_INCLUDE_GEO_NORMAL       "include_geometry_normal"
    #define MDL_CTX_OPTION_BUNDLE_RESOURCES         "bundle_resources"
    #define MDL_CTX_OPTION_EXPERIMENTAL             "experimental"

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

    template<typename T> 
    T get_option(const std::string& name) const {

        mi::Size index = get_option_index(name);
        ASSERT(M_SCENE, index < m_options.size());

        const Option& option = m_options[index];
        return STLEXT::any_cast<T> (option.get_value());
    }

    mi::Sint32 get_option(const std::string& name, STLEXT::Any& value) const;

    mi::Sint32 set_option(const std::string& name, const STLEXT::Any& value);

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

/// Outputs MDL messages to the logger.
///
/// Also adds the messages to \p context (unless \p context is \c NULL).
void report_messages(const mi::mdl::Messages& in_messages, Execution_context* context);

/// Wraps an MDL resource reader as IReader.
mi::neuraylib::IReader* get_reader( mi::mdl::IMDL_resource_reader* reader);

/// Creates an instance of an implementation of IMAGE::IMdr_callback.
IMAGE::IMdr_callback* create_mdr_callback();


// **********  Resource-related attributes *********************************************************

/// Retrieve the attributes of a texture resource.
///
/// \param transaction      The DB transaction to use.
/// \param texture          A texture resource.
/// \param[out] valid       The result of texture_isvalid().
/// \param[out] is_uvtile   True, if this is an uvtile texture.
/// \param[out] width       The width of \p texture.
/// \param[out] height      The height of \p texture.
/// \param[out] depth       The depth of \p texture.
///
/// \return \c true in case of success, \c false if texture is not a valid texture resource
bool get_texture_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* texture,
    bool& valid,
    bool& is_uvtile,
    int& width,
    int& height,
    int& depth);

/// Retrieve the resolution of given uvtile of a uvtile texture resource.
///
/// \param transaction      The DB transaction to use.
/// \param texture          A texture resource.
/// \param[int] uv_tile     The uvtile coordinates.
/// \param[out] width       The width of \p texture.
/// \param[out] height      The height of \p texture.
///
/// \return \c true in case of success, \c false if texture is not a valid uvtile texture resource
bool get_texture_uvtile_resolution(
    DB::Transaction* transaction,
    const mi::mdl::IValue* texture,
    mi::Sint32_2 const &uv_tile,
    int &width,
    int &height);

/// Retrieve the attributes of a light profile resource.
///
/// \param transaction       The DB transaction to use.
/// \param light_profile     A light profile resource.
/// \param[out] valid        The result of light_profile_isvalid().
/// \param[out] power        The result of light_profile_power().
/// \param[out] maximum      The result of light_profile_maximum().
///
/// \return \c true in case of success, \c false if light_profile is not a valid light profile
///         resource
bool get_light_profile_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* light_profile,
    bool& valid,
    float& power,
    float& maximum);

/// Retrieve the attributes of a BSDF measurement resource.
///
/// \param transaction       The DB transaction to use.
/// \param bsdf_measurement  A BSDF measurement resource.
/// \param[out] valid        The result of bsdf_measurement_isvalid().
///
/// \return \c true in case of success, \c false if light_profile is not a valid BSDF measurement
///         resource
bool get_bsdf_measurement_attributes(
    DB::Transaction* transaction,
    const mi::mdl::IValue* bsdf_measurement,
    bool& valid);


// **********  Mdl_dag_builder *********************************************************************

/// Helper class to handle conversion from MDL::IExpression into mi::mdl::DAG_node.
///
/// \tparam T   Either mi::mdl::IDag_builder or mi::mdl::IGenerated_code_dag::DAG_node_factory.
///
/// \note The class behaves differently for temporaries. If T == IDag_builder, temporaries are
///       re-converted each time they are encounterd. If T == IGenerated_code_dag::DAG_node_factory,
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
    /// \param mdl_meters_per_scene_unit   Conversion ratio between meters and scene units.
    /// \param mdl_wavelength_min          The smallest supported wavelength.
    /// \param mdl_wavelength_max          The largest supported wavelength.
    /// \param compiled_material           The compiled material that will be used to resolve
    ///                                    temporaries. Can be \c NULL if the expressions to be
    ///                                    converted do not contain any references to temporaries.
    Mdl_dag_builder(
        DB::Transaction* transaction,
        T* dag_builder,
        mi::Float32 mdl_meters_per_scene_unit,
        mi::Float32 mdl_wavelength_min,
        mi::Float32 mdl_wavelength_max,
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

    const mi::mdl::DAG_node* int_expr_list_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const Mdl_function_definition* function_definition,
        const char* function_definition_mdl_name,
        const char* function_call_name,
        const IExpression_list* arguments);

    const mi::mdl::DAG_node* int_expr_list_to_mdl_dag_node(
        const mi::mdl::IType* mdl_type,
        const Mdl_material_definition* material_definition,
        const char* material_definition_mdl_name,
        const char* material_call_name,
        const IExpression_list* arguments);

    /// Clones a DAG node.
    ///
    /// \param node  The DAG IR node to clone.
    /// \return      The clone of \p node.
    const mi::mdl::DAG_node* clone_dag_node(
        const mi::mdl::DAG_node* node);

private:
    /// The DB transaction to use (needed to access attached function calls or material instances).
    DB::Transaction* m_transaction;
    /// A DAG builder used to construct the DAG nodes.
    T* m_dag_builder;
    /// Conversion ratio between meters and scene units.
    mi::Float32 m_mdl_meters_per_scene_unit;
    /// The smallest supported wavelength.
    mi::Float32 m_mdl_wavelength_min;
    /// The largest supported wavelength.
    mi::Float32 m_mdl_wavelength_max;
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
    /// Current set of DAG calls in the current trace to check for cycles.
    std::set<MI::Uint32> m_call_trace;
};


// **********  Mdl_call_resolver *******************************************************************

/// Finds the owning module of a function definition.
class Mdl_call_resolver : public mi::mdl::ICall_name_resolver
{
public:
    /// Constructor.
    ///
    /// \param transaction  the transaction to use for name lookup
    Mdl_call_resolver( DB::Transaction* transaction) : m_transaction( transaction) { }

    virtual ~Mdl_call_resolver();

    /// Finds the owning module of a function definition.
    ///
    /// \param name   The MDL name of a function definition.
    /// \return       The owning module, or \c NULL in case of failures.
    const mi::mdl::IModule* get_owner_module( char const* name) const;

private:
    DB::Transaction* m_transaction;

    typedef std::set<const mi::mdl::IModule*> Module_set;
    mutable Module_set m_resolved_modules;
};

// **********  Resource names **********************************************************************

namespace DETAIL {

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
std::string unresolve_resource_filename(
    const char* filename, const char* module_filename, const char* module_name);

/// Returns an absolute MDL file path which can be used to reload the given resource with the
/// current search paths, or the empty string if this is not possible.
std::string unresolve_resource_filename(
    const char* archive_filename,
    const char* archive_membername,
    const char* module_filename,
    const char* module_name);

} // namespace DETAIL

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_UTILITIES_H

