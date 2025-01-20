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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_FUNCTION_DEFINITION_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_FUNCTION_DEFINITION_H

#include <tuple>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/imodule.h> // for mi::neuraylib::Mdl_version
#include <mi/neuraylib/ifunction_definition.h> // for mi::neuraylib::IFunction_definition::Semantic

#include <io/scene/scene/i_scene_scene_element.h>

#include "i_mdl_elements_module.h"
#include "i_mdl_elements_expression.h" // needed by Visual Studio

namespace mi { namespace mdl { class IGenerated_code_dag; class IType; } }

namespace MI {

namespace NEURAY { class Function_definition_impl; }

namespace MDL {

class Execution_context;
class IAnnotation_block;
class IAnnotation_list;
class IExpression;
class IExpression_factory;
class IExpression_list;
class IType;
class IType_factory;
class IType_list;
class IValue_factory;
class Mdl_function_call;

/// The class ID for the #Mdl_function_definition class.
static const SERIAL::Class_id ID_MDL_FUNCTION_DEFINITION = 0x5f4d6664; // '_Mfd'

class Mdl_function_definition
  : public SCENE::Scene_element<Mdl_function_definition, ID_MDL_FUNCTION_DEFINITION>
{
    friend class NEURAY::Function_definition_impl;
    friend class SCENE::Scene_element<Mdl_function_definition, ID_MDL_FUNCTION_DEFINITION>;
public:

    /// Constructor.
    ///
    /// \param transaction            The DB transaction to access the module and to resolve
    ///                               resources.
    /// \param function_tag           The tag this definition will eventually get (needed to pass
    ///                               on to function calls later).
    /// \param function_ident         The identifier of this definition will be used to check if it
    ///                               is still valid and has not been removed/altered due to a
    ///                               module reload.
    /// \param module                 The core module of this function definition.
    /// \param code_dag               The DAG representation of the MDL module.
    /// \param is_material            Indicates whether this is a material or a function. Note that
    ///                               the elemental constructor and copy constructor for material
    ///                               are functions, even though their return type is "material".
    /// \param function_index         The index of this definition in the module.
    /// \param module_filename        The filename of the module.
    /// \param module_mdl_name        The MDL module name.
    /// \param resolve_resources      \c  true, if resources are supposed to be loaded into the DB
    ///                               (if referenced in the parameter defaults).
    Mdl_function_definition(
        DB::Transaction* transaction,
        DB::Tag function_tag,
        Mdl_ident function_ident,
        const mi::mdl::IModule* module,
        const mi::mdl::IGenerated_code_dag* code_dag,
        bool is_material,
        mi::Size function_index,
        const char* module_filename,
        const char* module_mdl_name,
        bool resolve_resources);

    /// Copy constructor.
    Mdl_function_definition( const Mdl_function_definition&) = default;

    Mdl_function_definition& operator=( const Mdl_function_definition&) = delete;

    // methods corresponding to mi::neuraylib::IFunction_definition

    DB::Tag get_module( DB::Transaction* transaction) const;

    const char* get_mdl_name() const;

    const char* get_mdl_module_name() const;

    const char* get_mdl_simple_name() const;

    const char* get_mdl_parameter_type_name( mi::Size index) const;

    DB::Tag get_prototype() const;

    void get_mdl_version(
        mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const;

    mi::neuraylib::IFunction_definition::Semantics get_semantic() const;

    bool is_exported() const { return m_is_exported; }

    bool is_declarative() const { return m_is_declarative; }

    bool is_uniform() const { return m_is_uniform; }

    bool is_material() const { return m_is_material; }

    const IType* get_return_type() const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name( mi::Size index) const;

    mi::Size get_parameter_index( const char* name) const;

    const IType_list* get_parameter_types() const;

    const IExpression_list* get_defaults() const;

    const IExpression_list* get_enable_if_conditions() const;

    mi::Size get_enable_if_users( mi::Size index) const;

    mi::Size get_enable_if_user( mi::Size index, mi::Size u_index) const;

    const IAnnotation_block* get_annotations() const;

    const IAnnotation_block* get_return_annotations() const;

    const IAnnotation_list* get_parameter_annotations() const;

    const IExpression* get_body( DB::Transaction* transaction) const;

    mi::Size get_temporary_count( DB::Transaction* transaction) const;

    const IExpression* get_temporary( DB::Transaction* transaction, mi::Size index) const;

    const char* get_temporary_name( DB::Transaction* transaction, mi::Size index) const;

    std::string get_thumbnail() const;

    Mdl_function_call* create_function_call(
       DB::Transaction* transaction,
       const IExpression_list* arguments,
       mi::Sint32* errors = nullptr) const;

    std::string get_mangled_name( DB::Transaction* transaction) const;

    // internal methods

    /// The API method mi::neuraylib::IExpression_factory::create_direct_call() uses this method to
    /// do the actual work.
    IExpression_direct_call* create_direct_call(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// \name Methods used to implement create_function_call()
    //@{

    /// Implements #create_function_call() for the general case.
    Mdl_function_call* create_call_internal(
       DB::Transaction* transaction,
       const IExpression_list* arguments,
       bool allow_ek_parameter,
       bool immutable,
       mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the array constructor.
    Mdl_function_call* create_array_constructor_call_internal(
       DB::Transaction* transaction,
       const IExpression_list* arguments,
       bool allow_ek_parameter,
       bool immutable,
       mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the array index operator.
    Mdl_function_call* create_array_index_operator_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool immutable,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the array length operator.
    Mdl_function_call* create_array_length_operator_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool immutable,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the ternary operator.
    Mdl_function_call* create_ternary_operator_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool immutable,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the cast operator.
    Mdl_function_call* create_cast_operator_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool immutable,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_function_call() for the decl_cast operator.
    Mdl_function_call* create_decl_cast_operator_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool immutable,
        mi::Sint32* errors = nullptr) const;
    //@}

    /// \name Methods used to implement create_direct_call()
    //@{

    /// Implements #create_direct_call() for the general case.
    IExpression_direct_call* create_direct_call_internal(
       DB::Transaction* transaction,
       const IExpression_list* arguments,
       mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the array constructor.
    IExpression_direct_call* create_array_constructor_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the array index operator.
    IExpression_direct_call* create_array_index_operator_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the array length operator.
    IExpression_direct_call* create_array_length_operator_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the ternary operator.
    IExpression_direct_call* create_ternary_operator_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the cast operator.
    IExpression_direct_call* create_cast_operator_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    /// Implements #create_direct_call() for the decl_cast operator.
    IExpression_direct_call* create_decl_cast_operator_direct_call_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    //@}

    /// Returns the core semantic of this definition.
    mi::mdl::IDefinition::Semantics get_core_semantic() const;

    /// Returns the core DAG and the index of this definition.
    ///
    /// \note This function returns a pointer that is owned by the DB element for the corresponding
    ///        module. Therefore, DB elements of type MDL module are not flushable.
    const mi::mdl::IGenerated_code_dag* get_core_code_dag(
        DB::Transaction* transaction, mi::Size& definition_index) const;

    /// Returns the core type of an parameter.
    ///
    /// \note Does not support template-like functions.
    ///
    /// \note The return type is an owned interface, not a \em reference-counted interface.
    ///
    /// \note This function returns a pointer that is owned by the DB element for the corresponding
    ///        module. Therefore, DB elements of type MDL module are not flushable.
    const mi::mdl::IType* get_core_return_type( DB::Transaction* transaction) const;

    /// Returns the core return type.
    ///
    /// \note Does not support template-like functions.
    ///
    /// \note The return type is an owned interface, not a \em reference-counted interface.
    ///
    /// \note This function returns a pointer that is owned by the DB element for the corresponding
    ///       module. Therefore, DB elements of type MDL module are not flushable.
    ///
    /// \note Prefer #get_core_dag() to avoid repeated lookups for all parameter indices.
    const mi::mdl::IType* get_core_parameter_type(
        DB::Transaction* transaction, mi::Size index) const;

    /// Returns the MDL name without parameter types, i.e.,, the MDL module name plus "::" plus the
    /// simple name.
    std::string get_mdl_name_without_parameter_types() const;

    /// Returns the original function name (or \c nullptr if this definition is not re-exported).
    const char* get_mdl_original_name() const;

    /// Returns the database name of the module this definition belongs to.
    const char* get_module_db_name() const;

    /// Return a function hash if available.
    mi::base::Uuid get_function_hash() const { return m_function_hash; }

    /// Returns true if the definition still exists in the module.
    bool is_valid(
        DB::Transaction* transaction,
        Execution_context* context) const;

    /// Checks if this definition is compatible to the given definition.
    bool is_compatible(const Mdl_function_definition& other) const;

    /// Returns the identifier of this function definition.
    Mdl_ident get_ident() const;

    /// Computes the MDL versions in m_since_version and m_removed_version.
    ///
    /// m_since_version       The MDL version in which this function definition was added. If the
    ///                       function definition does not belong to the standard library, the
    ///                       MDL version of the module is returned.
    /// m_removed_version     The MDL version in which this function definition was removed, or
    ///                       mi::neuraylib::MDL_VERSION_INVALID if the function has not been
    ///                       removed so far or does not belong to the standard library.
    void compute_mdl_version( const mi::mdl::IModule* module);

    /// Improved version of SERIAL::Serializable::dump().
    ///

    /// \param transaction   The DB transaction (for name lookups and tag versions). Can be
    ///                      \c nullptr.
    void dump( DB::Transaction* transaction) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const { dump( /*transaction*/ nullptr); }

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const;

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

private:
    /// Default constructor.
    ///
    /// Does not create a valid instance, to be used by the deserializer only.
    Mdl_function_definition();

    /// \name Code shared between #create_function_call() and #create_direct_call().
    ///
    /// Fills in defaults if necessary, and insert casts if enabled and necessary.
    //@{

    /// Checks the arguments for call creation for the general case.
    IExpression_list* check_and_prepare_arguments(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool allow_ek_parameter,
        bool allow_ek_direct_call_and_temporaries,
        bool create_direct_calls,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the array constructor operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_array_constructor_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool allow_ek_parameter,
        bool allow_ek_direct_call_and_temporaries,
        bool create_direct_calls,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the array index operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_array_index_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the array length operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_array_length_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the ternary operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_ternary_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the cast operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_cast_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    /// Checks the arguments for call creation for the decl_cast operator.
    std::tuple<IExpression_list*,IType_list*,const IType*>
    check_and_prepare_arguments_decl_cast_operator(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool copy_immutable_calls,
        mi::Sint32* errors) const;

    //@}

    mi::base::Handle<IType_factory> m_tf;         ///< The type factory.
    mi::base::Handle<IValue_factory> m_vf;        ///< The value factory.
    mi::base::Handle<IExpression_factory> m_ef;   ///< The expression factory.

    std::string m_module_filename;                ///< The filename of the corr. module (or empty).
    std::string m_module_mdl_name;                ///< The MDL name of the corresponding module.
    std::string m_module_db_name;                 ///< The DB name of the corresponding module.
    DB::Tag m_function_tag;                       ///< The tag of this function definition.
    Mdl_ident m_function_ident;                   ///< The identifier of this function definition.
    mi::mdl::IDefinition::Semantics m_core_semantic;            ///< The semantic.
    mi::neuraylib::IFunction_definition::Semantics m_semantic;  ///< The semantic.
    std::string m_mdl_name;                       ///< The MDL name of this function definition.
    std::string m_simple_name;                    ///< The simple MDL name of this fct. definition.
    std::string m_db_name;                        ///< The DB name of this function definition.
    std::string m_original_name;                  ///< The original MDL function name (or empty).
    DB::Tag m_prototype_tag;                      ///< The prototype of this fct. def. (or invalid).
    bool m_is_exported;                           ///< The export flag.
    bool m_is_declarative;                        ///< The declarative flag.
    bool m_is_uniform;                            ///< The uniform flag.
    bool m_is_material;                           ///< Material or function
    mi::neuraylib::Mdl_version m_since_version;   ///< The version when this def. was added.
    mi::neuraylib::Mdl_version m_removed_version; ///< The version when this def. was removed.

    mi::base::Handle<IType_list> m_parameter_types;
    std::vector<std::string> m_parameter_type_names; ///< The MDL parameter type names.
    mi::base::Handle<const IType> m_return_type;
    mi::base::Handle<IExpression_list> m_defaults;
    mi::base::Handle<IAnnotation_block> m_annotations;
    mi::base::Handle<IAnnotation_list> m_parameter_annotations;
    mi::base::Handle<IAnnotation_block> m_return_annotations;
    mi::base::Handle<IExpression_list> m_enable_if_conditions;
    std::vector<std::vector<mi::Size>> m_enable_if_users;

    mi::base::Uuid m_function_hash;               ///< The function hash if any.

    /// Indicates whether resources are supposed to be loaded into the DB.
    bool m_resolve_resources;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_FUNCTION_DEFINITION_H
