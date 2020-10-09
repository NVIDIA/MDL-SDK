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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_DEFINITION_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_DEFINITION_H

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/imodule.h> // for mi::neuraylib::Mdl_version

#include <io/scene/scene/i_scene_scene_element.h>

#include "i_mdl_elements_expression.h" // needed by Visual Studio
#include "i_mdl_elements_module.h"

namespace mi { namespace mdl { class IGenerated_code_dag; class IType; } }

namespace MI {

namespace NEURAY { class Material_definition_impl; }

namespace MDL {

class Execution_context;
class IAnnotation_block;
class IAnnotation_list;
class IExpression;
class IExpression_factory;
class IExpression_list;
class IType_factory;
class IType;
class IType_list;
class IValue_factory;
class Mdl_material_instance;

/// The class ID for the #Mdl_material_definition class.
static const SERIAL::Class_id ID_MDL_MATERIAL_DEFINITION = 0x5f4d6d64; // '_Mmd'

class Mdl_material_definition
  : public SCENE::Scene_element<Mdl_material_definition, ID_MDL_MATERIAL_DEFINITION>
{
    friend class NEURAY::Material_definition_impl;
    friend class SCENE::Scene_element<Mdl_material_definition, ID_MDL_MATERIAL_DEFINITION>;
public:

    /// Constructor.
    ///
    /// \param transaction            The DB transaction to access the module and to resolved MDL
    ///                               resources.
    /// \param material_tag           The tag this definition will eventually get (needed to pass
    ///                               on to material instances calls later).
    /// \param material_ident         The identifier of this definition will be used to check if it
    ///                               is still valid and has not been removed/altered due to a module
    ///                               reload.
    /// \param module                 The MDL module of this material definition.
    /// \param code_dag               The DAG representation of the MDL module.
    /// \param material_index         The index of this definition in the module.
    /// \param module_filename        The filename of the module.
    /// \param module_name            The fully-qualified MDL module name.
    /// \param load_resources         True, if resources are supposed to be loaded into the DB.
    Mdl_material_definition(
        DB::Transaction* transaction,
        DB::Tag material_tag,
        Mdl_ident material_ident,
        const mi::mdl::IModule* module,
        const mi::mdl::IGenerated_code_dag* code_dag,
        mi::Size material_index,
        const char* module_filename,
        const char* module_name,
        bool load_resources);

    Mdl_material_definition& operator=( const Mdl_material_definition&) = delete;

    // methods corresponding to mi::neuraylib::IMaterial_definition

    DB::Tag get_module( DB::Transaction* transaction) const;

    const char* get_mdl_name() const;

    const char* get_mdl_module_name() const;

    const char* get_mdl_simple_name() const;

    DB::Tag get_prototype() const;

    void get_mdl_version(
        mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const;

    bool is_exported() const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name( mi::Size index) const;

    mi::Size get_parameter_index( const char* name) const;

    const IType_list* get_parameter_types() const;

    const IExpression_list* get_defaults() const;

    const IExpression_list* get_enable_if_conditions() const;

    mi::Size get_enable_if_users( mi::Size index) const;

    mi::Size get_enable_if_user( mi::Size index, mi::Size u_index) const;

    const IAnnotation_block* get_annotations() const;

    const IAnnotation_list* get_parameter_annotations() const;

    const IExpression_direct_call* get_body( DB::Transaction* transaction) const;

    mi::Size get_temporary_count( DB::Transaction* transaction) const;

    const IExpression* get_temporary( DB::Transaction* transaction, mi::Size index) const;

    const char* get_temporary_name( DB::Transaction* transaction, mi::Size index) const;

    const char* get_thumbnail() const;

    Mdl_material_instance* create_material_instance(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        mi::Sint32* errors = nullptr) const;

    // internal methods

    /// Internal variant of #create_material_instance().
    ///
    /// \param allow_ek_parameter If set to \c true, expressions of type EK_PARAMETER are also
    ///                           permitted as arguments. This flag may only be set by the MDL
    ///                           integration itself, not by external callers.
    /// \param immutable          If set to \c true, the created material instance is flagged as
    ///                           immutable.
    Mdl_material_instance* create_material_instance_internal(
        DB::Transaction* transaction,
        const IExpression_list* arguments,
        bool allow_ek_parameter,
        bool immutable,
        mi::Sint32* errors = nullptr) const;

    /// Returns the MDL type of an parameter.
    ///
    /// \note The return type is an owned interface, not a \em reference-counted interface.
    ///
    /// \note This function returns a pointer that is owned by the DB element for the corresponding
    ///        module. Therefore, DB elements of type MDL module are not flushable.
    const mi::mdl::IType* get_mdl_parameter_type(
        DB::Transaction* transaction, mi::Uint32 index) const;

    /// Returns the MDL name without parameter types, i.e.,, the MDL module name plus "::" plus the
    /// simple name.
    ///
    /// For materials, this is identical to the MDL name itself. This function exists only to
    /// simplify template code.
    std::string get_mdl_name_without_parameter_types() const { return get_mdl_name(); }

    /// Returns the original material name (or \c NULL if this definition is not re-exported).
    const char* get_mdl_original_name() const;

    /// Returns the database name of the module this definition belongs to.
    const char* get_module_db_name() const;

    /// Returns true if this definition still exists in the module.
    bool is_valid(DB::Transaction* transaction, Execution_context *context) const;

    /// Checks if this definition is compatible to the given definition.
    bool is_compatible(const Mdl_material_definition& other) const;

    /// Returns the identifier of this material definition.
    Mdl_ident get_ident() const;

    /// Returns the MDL version when this material definition was added and removed.
    ///
    /// \param[out] since     The MDL version in which this material definition was added. Since
    ///                       there are no material definitions in the standard library, the
    ///                       MDL version of the corresponding module is returned.
    /// \param[out] removed   The MDL version in which this material definition was removed. Since
    ///                       there are no material definitions in the standard library,
    ///                       mi_mdl_IMDL_MDL_VERSION_INVALID is always returned.
    void get_mdl_version(
        mi::mdl::IMDL::MDL_version& since, mi::mdl::IMDL::MDL_version& removed) const;

    /// Improved version of SERIAL::Serializable::dump().
    ///
    /// \param transaction   The DB transaction (for name lookups and tag versions). Can be \c NULL.
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
    Mdl_material_definition();

private:

    mi::base::Handle<IType_factory> m_tf;        ///< The type factory.
    mi::base::Handle<IValue_factory> m_vf;       ///< The value factory.
    mi::base::Handle<IExpression_factory> m_ef;  ///< The expression factory.

    std::string m_module_mdl_name;               ///< The MDL name of the corresponding module.
    std::string m_module_db_name;                ///< The DB name of the corresponding module.
    DB::Tag m_material_tag;                      ///< The tag of this material definition.
    Mdl_ident m_material_ident;                  ///< The identifier of this material definition.
    std::string m_name;                          ///< The MDL name of the material definition.
    std::string m_simple_name;                   ///< The simple MDL name of this material definition.
    std::string m_db_name;                       ///< The DB name of the material definition.
    std::string m_original_name;                 ///< The original MDL function name (or empty).
    mutable std::string m_thumbnail;             ///< The thumbnail image for this definition.
    DB::Tag m_prototype_tag;                     ///< The prototype of this mat. def. (or inv. tag).
    bool m_is_exported;                          ///< The export flag.
    mi::mdl::IMDL::MDL_version m_since_version;   ///< The version when this def. was added.
    mi::mdl::IMDL::MDL_version m_removed_version; ///< The version when this def. was removed.

    mi::base::Handle<IType_list> m_parameter_types;
    mi::base::Handle<IExpression_list> m_defaults;
    mi::base::Handle<IAnnotation_block> m_annotations;
    mi::base::Handle<IAnnotation_list> m_parameter_annotations;
    mi::base::Handle<IExpression_list> m_enable_if_conditions;
    std::vector< std::vector<mi::Sint32> > m_enable_if_users;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_MATERIAL_DEFINITION_H
