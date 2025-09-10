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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_COMPILED_MATERIAL_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_COMPILED_MATERIAL_H

#include <io/scene/scene/i_scene_scene_element.h>

#include <mi/base/handle.h>
#include <mi/base/uuid.h>
#include <mi/neuraylib/icompiled_material.h>

#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>

#include <string>
#include <set>

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_resource_tag_tuple.h"
#include "i_mdl_elements_module.h"

namespace mi { namespace mdl { class IMaterial_instance; } }

namespace MI {

namespace MDL {

class IExpression;
class IExpression_factory;
class IExpression_list;
class IType_factory;
class IValue;
class IValue_factory;
class IValue_list;
class Mdl_dag_converter;

/// The class ID for the #Mdl_compiled_material class.
static const SERIAL::Class_id ID_MDL_COMPILED_MATERIAL = 0x5f436d69; // '_Cmi'

class Mdl_compiled_material
  : public SCENE::Scene_element<Mdl_compiled_material, ID_MDL_COMPILED_MATERIAL>
{
public:
    /// Default constructor.
    ///
    /// Does not create a valid instance, to be used by the deserializer only.
    Mdl_compiled_material();

    /// Constructor.
    ///
    /// \param transaction                 The DB transaction to use.
    /// \param core_material_instance      The core material instance.
    /// \param module_tag                  Tag of the MDL module containing the underlying function
    ///                                    definition. Optional (not available for distilled
    ///                                    materials).
    /// \param mdl_meters_per_scene_unit   Conversion ratio between meters and scene units.
    /// \param mdl_wavelength_min          The smallest supported wavelength.
    /// \param mdl_wavelength_max          The largest supported wavelength.
    /// \param resolve_resources           \c true if resources are supposed to be loaded into the
    ///                                    DB.
    Mdl_compiled_material(
        DB::Transaction* transaction,
        const mi::mdl::IMaterial_instance* core_material_instance,
        DB::Tag module_tag,
        mi::Float32 mdl_meters_per_scene_unit,
        mi::Float32 mdl_wavelength_min,
        mi::Float32 mdl_wavelength_max,
        bool resolve_resources);

    /// Copy constructor.
    Mdl_compiled_material( const Mdl_compiled_material&) = default;

    Mdl_compiled_material& operator=( const Mdl_compiled_material&) = delete;

    // methods corresponding to mi::neuraylib::ICompiled_material

    const IExpression_direct_call* get_body( DB::Transaction* transaction) const;

    mi::Size get_temporary_count() const;

    const IExpression* get_temporary( DB::Transaction* transaction, mi::Size index) const;

    const IExpression* lookup_sub_expression( DB::Transaction* transaction, const char* path) const;

    bool is_valid( DB::Transaction* transaction, Execution_context* context) const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name( mi::Size index) const;

    const IValue* get_argument( DB::Transaction* transaction, mi::Size index) const;

    DB::Tag get_connected_function_db_name(
        DB::Transaction* transaction,
        DB::Tag material_instance_tag,
        mi::Size parameter_index) const;

    mi::Float32 get_mdl_meters_per_scene_unit() const { return m_mdl_meters_per_scene_unit; }

    mi::Float32 get_mdl_wavelength_min() const { return m_mdl_wavelength_min; }

    mi::Float32 get_mdl_wavelength_max() const { return m_mdl_wavelength_max; }

    mi::mdl::IMaterial_instance::Opacity get_opacity() const;

    mi::mdl::IMaterial_instance::Opacity get_surface_opacity() const;

    bool get_cutout_opacity( mi::Float32* cutout_opacity) const;

    mi::Size get_referenced_scene_data_count() const;

    const char* get_referenced_scene_data_name( mi::Size index) const;

    bool depends_on_state_transform() const;

    bool depends_on_state_object_id() const;

    bool depends_on_global_distribution() const;

    bool depends_on_uniform_scene_data() const;

    mi::base::Uuid get_hash() const;

    mi::base::Uuid get_slot_hash( mi::neuraylib::Material_slot slot) const;

    mi::base::Uuid get_sub_expression_hash( const char* path) const;

    // internal methods

    /// Returns the wrapped core material instance.
    const mi::mdl::IMaterial_instance* get_core_material_instance() const;

    /// Returns the internal space.
    const char* get_internal_space() const;

    /// Returns the number of resource table entries.
    mi::Size get_resources_count() const;

    /// Returns the \p index -th resource table entry.
    const Resource_tag_tuple* get_resource_tag_tuple( mi::Size index) const;

    /// Returns the "resolve_resources" setting from the constructor.
    ///
    /// Used by the distiller to pass that setting to the distilled instance.
    bool get_resolve_resources() const { return m_resolve_resources; }

     /// Returns all arguments as one value list.
     const IValue_list* get_arguments( DB::Transaction* transaction) const;

    /// Swaps *this and \p other.
    ///
    /// Used by the API to move the content of just constructed DB elements into the already
    /// existing API wrapper.
    void swap( Mdl_compiled_material& other);

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const;

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

private:
    /// Returns an instance of Mdl_dag_converter configured in a way suitable for converting body,
    /// temporaries and arguments of this compiled material.
    std::unique_ptr<Mdl_dag_converter> get_dag_converter( DB::Transaction* transaction) const;

    mi::base::Handle<IType_factory> m_tf;             ///< The type factory.
    mi::base::Handle<IValue_factory> m_vf;            ///< The value factory.
    mi::base::Handle<IExpression_factory> m_ef;       ///< The expression factory.

    /// The wrapped core material instance.
    mi::base::Handle<const mi::mdl::IMaterial_instance> m_core_material_instance;

    std::vector<Resource_tag_tuple> m_resources;      ///< The resources.

    mi::Float32 m_mdl_meters_per_scene_unit = 1.0f;   ///< The conversion ratio.
    mi::Float32 m_mdl_wavelength_min = 0.0f;          ///< The smallest supported wavelength.
    mi::Float32 m_mdl_wavelength_max = 0.0f;          ///< The largest supported wavelength.

    /// Indicates whether resources are supposed to be loaded into the DB.
    bool m_resolve_resources = false;

    /// The set of all referenced tags (function calls and resources).
    DB::Tag_set m_tags;

    /// Module identifiers of all remaining call expressions plus the module given in the
    /// constructor.
    ///
    /// Used by is_valid().
    ///
    /// TODO Just considering the remaining call expressions is wrong. We need to consider all used
    /// call expressions in the input (before optimization), including their imports.
    std::set<Mdl_tag_ident> m_module_idents;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_COMPILED_MATERIAL_H
