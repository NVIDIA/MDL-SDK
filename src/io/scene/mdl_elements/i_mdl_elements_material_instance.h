/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_INSTANCE_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_INSTANCE_H

#include <mi/base/handle.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <io/scene/scene/i_scene_scene_element.h>

#include "i_mdl_elements_type.h" // needed by Visual Studio
#include "i_mdl_elements_module.h"

namespace mi { namespace mdl { class IType; } }

namespace MI {

namespace MDL {

class Execution_context;
class IExpression;
class IExpression_factory;
class IExpression_list;
class IType_factory;
class IType_list;
class IValue_factory;
class Mdl_compiled_material;

/// The class ID for the #Mdl_material_instance class.
static const SERIAL::Class_id ID_MDL_MATERIAL_INSTANCE = 0x5f4d6d69; // '_Mmi'

class Mdl_material_instance
  : public SCENE::Scene_element<Mdl_material_instance, ID_MDL_MATERIAL_INSTANCE>
{
public:

    /// Default constructor.
    ///
    /// Does not create a valid instance, to be used by the deserializer only.
    Mdl_material_instance();

    /// Constructor.
    Mdl_material_instance(
        DB::Tag module_tag,
        const char* module_db_name,
        DB::Tag definition_tag,
        Mdl_ident definition_ident,
        IExpression_list* arguments,
        const char* definition_name,
        const IType_list* parameter_types,
        bool immutable,
        const IExpression_list* enable_if_conditions);

    /// Copy constructor.
    Mdl_material_instance( const Mdl_material_instance& other);

    Mdl_material_instance& operator=( const Mdl_material_instance&) = delete;

    // methods corresponding to mi::neuraylib::IMaterial_instance

    DB::Tag get_material_definition(DB::Transaction *transaction) const;

    const char* get_mdl_material_definition() const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name( mi::Size index) const;

    mi::Size get_parameter_index( const char* name) const;

    const IType_list* get_parameter_types() const;

    const IExpression_list* get_arguments() const;

    mi::Sint32 set_arguments( DB::Transaction* transaction, const IExpression_list* arguments);

    mi::Sint32 set_argument(
        DB::Transaction* transaction, mi::Size index, const IExpression* argument);

    mi::Sint32 set_argument(
        DB::Transaction* transaction, const char* name, const IExpression* argument);

    Mdl_compiled_material* create_compiled_material(
        DB::Transaction* transaction,
        bool class_compilation,
        Execution_context* context) const;

    // internal methods

    /// Indicates whether the material instance is immutable.
    bool is_immutable() const { return m_immutable; }

    /// Makes the material instance mutable.
    ///
    /// This method may only be set to \c true by the MDL integration itself, not by external
    /// callers.
    void make_mutable(DB::Transaction* transaction);

    /// Creates a DAG material instance for this DB material instance (used by the DB compiled
    /// material instance and an utility function).
    ///
    /// \param transaction                 The transaction.
    /// \param use_temporaries             Indicates whether temporaries are used to represent
    ///                                    common subexpressions.
    /// \param class_compilation           Flag that selects class compilation instead of instance
    ///                                    compilation.
    /// \param mdl_meters_per_scene_unit   Conversion ratio between meters and scene units.
    /// \param mdl_wavelength_min          The smallest supported wavelength.
    /// \param mdl_wavelength_max          The largest supported wavelength.
    /// \param errors                      An optional pointer to an mi::Sint32 to which an error
    ///                                    code will be written. The error codes have the following
    ///                                    meaning:
    ///                                    -  0: Success.
    ///                                    - -1: An argument of the material instance has an
    ///                                          incorrect type.
    ///                                    - -2: The thin-walled material instance has different
    ///                                          transmission for surface and backface.
    ///                                    - -3: An varying argument was attached to an uniform
    ///                                          parameter.
    /// \return                            The DAG material instance, or \c NULL in case of failure.
    const mi::mdl::IGenerated_code_dag::IMaterial_instance* create_dag_material_instance(
        DB::Transaction* transaction,
        bool use_temporaries,
        bool class_compilation,
        Execution_context* context) const;

    /// Returns the MDL type of an parameter.
    ///
    /// \note The return type is an owned interface, not a \em reference-counted interface.
    const mi::mdl::IType* get_mdl_parameter_type(
        DB::Transaction* transaction, mi::Uint32 index) const;

    /// Returns the identifier of the definition.
    Mdl_ident get_definition_ident() const { return m_definition_ident; }

    /// Returns the DB name of the definition.
    const char* get_definition_db_name() const { return m_definition_db_name.c_str(); }

    /// Swaps *this and \p other.
    ///
    /// Used by the API to move the content of just constructed DB elements into the already
    /// existing API wrapper.
    void swap( Mdl_material_instance& other);

    /// Improved version of SERIAL::Serializable::dump().
    ///
    /// \param transaction   The DB transaction (for name lookups and tag versions). Can be \c NULL.
    void dump( DB::Transaction* transaction) const;

    /// Get the list of enable_if conditions.
    const IExpression_list* get_enable_if_conditions() const;

    /// Checks, if the material and its arguments still refer to valid definitions.
    bool is_valid(
        DB::Transaction* transaction,
        Execution_context* context) const;

    /// Checks, if the material and its arguments still refer to valid definitions.
    bool is_valid(
        DB::Transaction* transaction,
        DB::Tag_set& tags_seen,
        Execution_context* context) const;

    /// Attempts to repair an invalid material instance by trying to promote its definition
    /// tag identifier.
    /// \param transaction              the DB transaction.
    /// \param repair_invalid_calls     \c true, if invalid calls should be removed.
    /// \param remove_invalid_calls     \c true, if invalid calls should be repaired.
    /// \param level                    the recursion level.
    /// \param context                  will receive error messages.
    /// \return
    ///         -  0: Success.
    ///         - -1: Failure. Consult the context for details.
    mi::Sint32 repair(
        DB::Transaction* transaction,
        bool repair_invalid_calls,
        bool remove_invalid_calls,
        mi::Uint32 level,
        Execution_context* context);

    DB::Tag get_module() const;

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

    mi::base::Handle<IType_factory> m_tf;        ///< The type factory.
    mi::base::Handle<IValue_factory> m_vf;       ///< The value factory.
    mi::base::Handle<IExpression_factory> m_ef;  ///< The expression factory.

    // Members marked with (*) are duplicated from the corresponding material definition to avoid
    // frequent DB accesses.

    DB::Tag m_module_tag;                        ///< The corresponding MDL module. (*)
    DB::Tag m_definition_tag;                    ///< The corresponding material definition.
    Mdl_ident m_definition_ident;                ///< The corresponding material definition identifier.
    std::string m_module_db_name;                ///< The DB name of the module. (*)
    std::string m_definition_name;               ///< The MDL name of the material definition. (*)
    std::string m_definition_db_name;            ///< The DB name of the material definition. (*)
    bool m_immutable;                            ///< The immutable flag (set for defaults).

    mi::base::Handle<const IType_list> m_parameter_types;            // (*)
    mi::base::Handle<IExpression_list> m_arguments;

    mi::base::Handle<const IExpression_list> m_enable_if_conditions; // (*)
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_INSTANCE_H
