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

#include "pch.h"

#include "i_mdl_elements_compiled_material.h"

#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_module.h"
#include "mdl_elements_utilities.h"

#include <sstream>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/registry/i_config_registry.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/db/i_db_access.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>


namespace MI {

namespace MDL {

Mdl_compiled_material::Mdl_compiled_material()
  : m_hash( mi::base::Uuid{ 0, 0, 0, 0 }),
    m_mdl_meters_per_scene_unit( 1.0f),   // avoid warning
    m_mdl_wavelength_min( 0.0f),
    m_mdl_wavelength_max( 0.0f),
    m_properties( 0),  // avoid ubsan warning with swap() and temporaries
    m_opacity( mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN),
    m_surface_opacity( mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN),
    m_cutout_opacity( -1.0f),
    m_has_cutout_opacity( false)
{
    m_tf = get_type_factory();
    m_vf = get_value_factory();
    m_ef = get_expression_factory();

    memset( m_slot_hashes, 0, sizeof( m_slot_hashes));
}

Mdl_compiled_material::Mdl_compiled_material(
    DB::Transaction* transaction,
    const mi::mdl::IGenerated_code_dag::IMaterial_instance* instance,
    const char* module_name,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max,
    bool resolve_resources)
  : m_tf( get_type_factory()),
    m_vf( get_value_factory()),
    m_ef( get_expression_factory()),
    m_mdl_meters_per_scene_unit( mdl_meters_per_scene_unit),
    m_mdl_wavelength_min( mdl_wavelength_min),
    m_mdl_wavelength_max( mdl_wavelength_max),
    m_properties( instance->get_properties()),
    m_internal_space( instance->get_internal_space()),
    m_opacity( mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN),
    m_cutout_opacity( -1.0f),
    m_has_cutout_opacity( false)
{
    // Do *not* pass module_name to the DAG converter. This is only necessary for localization of
    // annnotations, which does matter here. Avoids misuse of this information for other (wrong)
    // purposes.
    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        instance->get_resource_tagger(),
        /*code_dag*/ nullptr,
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_name*/ nullptr,
        /*prototype_tag*/ DB::Tag(),
        resolve_resources,
        &m_module_idents);

    const mi::mdl::DAG_call* constructor = instance->get_constructor();
    mi::base::Handle<IExpression> body(
        converter.mdl_dag_node_to_int_expr( constructor, /*type_int*/ nullptr));
    ASSERT( M_SCENE, body);
    m_body = body->get_interface<IExpression_direct_call>();
    ASSERT( M_SCENE, m_body);

    mi::Size n = instance->get_temporary_count();
    m_temporaries = m_ef->create_expression_list( n);
    for( mi::Size i = 0; i < n; ++i) {
        std::string name( std::to_string( i));
        const mi::mdl::DAG_node* mdl_temporary = instance->get_temporary_value( i);
        mi::base::Handle<const IExpression> temporary(
            converter.mdl_dag_node_to_int_expr( mdl_temporary, /*type_int*/ nullptr));
        ASSERT( M_SCENE, temporary);
        m_temporaries->add_expression_unchecked( name.c_str(), temporary.get());
    }

    n = instance->get_parameter_count();
    m_arguments = m_vf->create_value_list( n);
    for( mi::Size i = 0; i < n; ++i) {
        const char* name = instance->get_parameter_name( i);
        const mi::mdl::IValue* mdl_argument = instance->get_parameter_default( i);
        mi::base::Handle<const IValue> argument(
            converter.mdl_value_to_int_value( /*type_int*/ nullptr, mdl_argument));
        ASSERT( M_SCENE, argument);
        m_arguments->add_value_unchecked( name, argument.get());
    }

    const mi::mdl::DAG_hash* h = instance->get_hash();
    m_hash = convert_hash( *h);

    for( int i = 0; i <= mi::mdl::IGenerated_code_dag::IMaterial_instance::MS_LAST; ++i) {
        h = instance->get_slot_hash(
            static_cast<mi::mdl::IGenerated_code_dag::IMaterial_instance::Slot>( i));
        m_slot_hashes[i] = convert_hash( *h);
    }

    for( size_t i = 0, n = instance->get_referenced_scene_data_count(); i < n; ++i) {
        m_referenced_scene_data.emplace_back( instance->get_referenced_scene_data_name( i));
    }

    m_opacity = instance->get_opacity();
    m_surface_opacity = instance->get_surface_opacity();
    m_has_cutout_opacity = false;
    if( const mi::mdl::IValue_float* v_cutout = instance->get_cutout_opacity()) {
        m_has_cutout_opacity = true;
        m_cutout_opacity = v_cutout->get_value();
    }

    if( module_name) {
        DB::Tag module_tag = transaction->name_to_tag( get_db_name( module_name).c_str());
        DB::Access<Mdl_module> module( module_tag, transaction);
        m_module_idents.insert( Mdl_tag_ident( module_tag, module->get_ident()));
    }

    // copy the resource tag table
    for( size_t i = 0, n = instance->get_resource_tag_map_entries_count(); i < n; ++i) {
        const mi::mdl::Resource_tag_tuple* e = instance->get_resource_tag_map_entry( i);
        m_resources.push_back( Resource_tag_tuple( *e));
    }
}

const IExpression_direct_call* Mdl_compiled_material::get_body() const
{
    m_body->retain();
    return m_body.get();
}

mi::Size Mdl_compiled_material::get_temporary_count() const
{
    return m_temporaries->get_size();
}

const IExpression* Mdl_compiled_material::get_temporary( mi::Size index) const
{
    return m_temporaries->get_expression( index);
}

mi::Float32 Mdl_compiled_material::get_mdl_meters_per_scene_unit() const
{
    return m_mdl_meters_per_scene_unit;
}

mi::Float32 Mdl_compiled_material::get_mdl_wavelength_min() const
{
    return m_mdl_wavelength_min;
}

mi::Float32 Mdl_compiled_material::get_mdl_wavelength_max() const
{
    return m_mdl_wavelength_max;
}

bool Mdl_compiled_material::depends_on_state_transform() const
{
    return 0 !=
        (m_properties & mi::mdl::IGenerated_code_dag::IMaterial_instance::IP_DEPENDS_ON_TRANSFORM);
}

bool Mdl_compiled_material::depends_on_state_object_id() const
{
    return 0 !=
        (m_properties & mi::mdl::IGenerated_code_dag::IMaterial_instance::IP_DEPENDS_ON_OBJECT_ID);
}

bool Mdl_compiled_material::depends_on_global_distribution() const
{
    return 0 !=
        (m_properties &
         mi::mdl::IGenerated_code_dag::IMaterial_instance::IP_DEPENDS_ON_GLOBAL_DISTRIBUTION);
}

bool Mdl_compiled_material::depends_on_uniform_scene_data() const
{
    return 0 !=
        (m_properties &
            mi::mdl::IGenerated_code_dag::IMaterial_instance::IP_DEPENDS_ON_UNIFORM_SCENE_DATA);
}

mi::Size Mdl_compiled_material::get_referenced_scene_data_count() const
{
    return m_referenced_scene_data.size();
}

const char* Mdl_compiled_material::get_referenced_scene_data_name( mi::Size index) const
{
    if( index < m_referenced_scene_data.size())
        return m_referenced_scene_data[index].c_str();
    return nullptr;
}

mi::Size Mdl_compiled_material::get_parameter_count() const
{
    return m_arguments->get_size();
}

const char* Mdl_compiled_material::get_parameter_name( mi::Size index) const
{
    return m_arguments->get_name( index);
}

const IValue* Mdl_compiled_material::get_argument( mi::Size index) const
{
    return m_arguments->get_value( index);
}

mi::base::Uuid Mdl_compiled_material::get_hash() const
{
    return m_hash;
}

mi::base::Uuid Mdl_compiled_material::get_slot_hash( mi::neuraylib::Material_slot slot) const
{
    using T = mi::mdl::IGenerated_code_dag::IMaterial_instance;

    switch( slot) {
        case mi::neuraylib::SLOT_THIN_WALLED:
            return m_slot_hashes[T::MS_THIN_WALLED];
        case mi::neuraylib::SLOT_SURFACE_SCATTERING:
            return m_slot_hashes[T::MS_SURFACE_BSDF_SCATTERING];
        case mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION:
            return m_slot_hashes[T::MS_SURFACE_EMISSION_EDF_EMISSION];
        case mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY:
            return m_slot_hashes[T::MS_SURFACE_EMISSION_INTENSITY];
        case mi::neuraylib::SLOT_SURFACE_EMISSION_MODE:
            return m_slot_hashes[T::MS_SURFACE_EMISSION_MODE];
        case mi::neuraylib::SLOT_BACKFACE_SCATTERING:
            return m_slot_hashes[T::MS_BACKFACE_BSDF_SCATTERING];
        case mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION:
            return m_slot_hashes[T::MS_BACKFACE_EMISSION_EDF_EMISSION];
        case mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY:
            return m_slot_hashes[T::MS_BACKFACE_EMISSION_INTENSITY];
        case mi::neuraylib::SLOT_BACKFACE_EMISSION_MODE:
            return m_slot_hashes[T::MS_BACKFACE_EMISSION_MODE];
        case mi::neuraylib::SLOT_IOR:
            return m_slot_hashes[T::MS_IOR];
        case mi::neuraylib::SLOT_VOLUME_SCATTERING:
            return m_slot_hashes[T::MS_VOLUME_VDF_SCATTERING];
        case mi::neuraylib::SLOT_VOLUME_ABSORPTION_COEFFICIENT:
            return m_slot_hashes[T::MS_VOLUME_ABSORPTION_COEFFICIENT];
        case mi::neuraylib::SLOT_VOLUME_SCATTERING_COEFFICIENT:
            return m_slot_hashes[T::MS_VOLUME_SCATTERING_COEFFICIENT];
        case mi::neuraylib::SLOT_VOLUME_EMISSION_INTENSITY:
            return m_slot_hashes[T::MS_VOLUME_EMISSION_INTENSITY];
        case mi::neuraylib::SLOT_GEOMETRY_DISPLACEMENT:
            return m_slot_hashes[T::MS_GEOMETRY_DISPLACEMENT];
        case mi::neuraylib::SLOT_GEOMETRY_CUTOUT_OPACITY:
            return m_slot_hashes[T::MS_GEOMETRY_CUTOUT_OPACITY];
        case mi::neuraylib::SLOT_GEOMETRY_NORMAL:
            return m_slot_hashes[T::MS_GEOMETRY_NORMAL];
        case mi::neuraylib::SLOT_HAIR:
            return m_slot_hashes[T::MS_HAIR];
        case mi::neuraylib::SLOT_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
        return mi::base::Uuid();
    }

    ASSERT( M_SCENE, false);
    return mi::base::Uuid();
}

const IExpression* Mdl_compiled_material::lookup_sub_expression( const char* path) const
{
    ASSERT( M_SCENE, path);

    return MDL::lookup_sub_expression( m_ef.get(), m_temporaries.get(), m_body.get(), path);
}

namespace {

DB::Tag get_next_call(const Mdl_function_call* mdl_instance, const std::string& parameter_name)
{
    mi::Size p_index = mdl_instance->get_parameter_index(parameter_name.c_str());
    if (p_index == static_cast<mi::Size>(-1))
        return DB::Tag();

    mi::base::Handle<const IExpression_list> arguments(mdl_instance->get_arguments());
    mi::base::Handle<const IExpression> expr(arguments->get_expression(p_index));
    mi::base::Handle<const IExpression_call> expr_call(
        expr->get_interface<const IExpression_call>());
    if (!expr_call)
        return DB::Tag();

    return expr_call->get_call();
}

} // namespace

DB::Tag Mdl_compiled_material::get_connected_function_db_name(
    DB::Transaction* transaction,
    DB::Tag material_instance_tag,
    const std::string& parameter_name) const
{
    DB::Access<Mdl_function_call> material_instance(material_instance_tag, transaction);
    ASSERT(M_SCENE, material_instance.is_valid());

    std::vector <std::string> path_tokens;
    boost::split(path_tokens, parameter_name, boost::is_any_of("."));

    // There needs to be at least one item, last one is the param name.
    // For struct constructors attached to the material, there is only one token.
    // For other attached functions there are more.
    if (path_tokens.size() == 0)
        return DB::Tag();

    DB::Tag call_tag = material_instance_tag;
    for (std::size_t i = 0; i < path_tokens.size(); ++i)
    {
        SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
        if (class_id == MDL::ID_MDL_FUNCTION_CALL) {

            DB::Access<Mdl_function_call> mdl_instance(call_tag, transaction);
            ASSERT(M_SCENE, mdl_instance.is_valid());

            DB::Tag next_tag = get_next_call(mdl_instance.get_ptr(), path_tokens[i]);
            if (next_tag.is_invalid())
                break;

            call_tag = next_tag;
            continue;
        }
        else {
            ASSERT(M_SCENE, false);
            return DB::Tag();
        }
    }
    // if no function is attached, the zero-tag is returned
    return call_tag == material_instance_tag ? DB::Tag() : call_tag;
}

mi::mdl::IGenerated_code_dag::IMaterial_instance::Opacity
Mdl_compiled_material::get_opacity() const
{
    return m_opacity;
}

mi::mdl::IGenerated_code_dag::IMaterial_instance::Opacity
Mdl_compiled_material::get_surface_opacity() const
{
    return m_surface_opacity;
}

bool Mdl_compiled_material::get_cutout_opacity( mi::Float32* cutout_opacity) const
{
    if( cutout_opacity) {
        *cutout_opacity = m_cutout_opacity;
        return m_has_cutout_opacity;
    }
    return false;
}

bool Mdl_compiled_material::is_valid(
    DB::Transaction* transaction, Execution_context* context) const
{
    for( const auto& id : m_module_idents) {

        DB::Access<Mdl_module> module( id.first, transaction);
        if( module->get_ident() != id.second) {
            std::string message = "The identifier of the imported module '"
                + get_db_name( module->get_mdl_name())
                + "' has changed.";
            add_error_message( context, message, -1);
            return false;
        }
        if( !module->is_valid( transaction, context))
            return false;
    }

    return true;
}

const char* Mdl_compiled_material::get_internal_space() const
{
    return m_internal_space.c_str();
}

const IExpression_list* Mdl_compiled_material::get_temporaries() const
{
    m_temporaries->retain();
    return m_temporaries.get();
}

const IValue_list* Mdl_compiled_material::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

mi::Size Mdl_compiled_material::get_resource_entries_count() const
{
    return m_resources.size();
}

const Resource_tag_tuple* Mdl_compiled_material::get_resource_entry( mi::Size index) const
{
    if( index >= m_resources.size())
        return nullptr;

    return &m_resources[index];
}

void Mdl_compiled_material::swap( Mdl_compiled_material& other)
{
    SCENE::Scene_element<Mdl_compiled_material, ID_MDL_COMPILED_MATERIAL>::swap(
        other);

    m_body.swap( other.m_body);
    m_temporaries.swap( other.m_temporaries);
    m_arguments.swap( other.m_arguments);
    m_resources.swap( other.m_resources);

    std::swap( m_hash, other.m_hash);
    for( int i = 0; i < mi::mdl::IGenerated_code_dag::IMaterial_instance::MS_LAST+1; ++i)
        std::swap( m_slot_hashes[i], other.m_slot_hashes[i]);
    std::swap( m_mdl_meters_per_scene_unit, other.m_mdl_meters_per_scene_unit);
    std::swap( m_mdl_wavelength_min, other.m_mdl_wavelength_min);
    std::swap( m_mdl_wavelength_max, other.m_mdl_wavelength_max);
    std::swap( m_properties, other.m_properties);
    std::swap( m_referenced_scene_data, other.m_referenced_scene_data);
    std::swap( m_internal_space, other.m_internal_space);
    std::swap( m_opacity, other.m_opacity);
    std::swap( m_surface_opacity, other.m_surface_opacity);
    std::swap( m_cutout_opacity, other.m_cutout_opacity);
    std::swap( m_has_cutout_opacity, other.m_has_cutout_opacity);
    std::swap( m_module_idents, other.m_module_idents);
}

namespace {

const char* get_opacity_str( mi::mdl::IGenerated_code_dag::IMaterial_instance::Opacity opacity)
{
    switch( opacity) {
        case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE:
            return "opaque";
        case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_TRANSPARENT:
            return "transparent";
        case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN:
            return "unknown";
    }

    ASSERT( M_SCENE, false);
    return "unknown";
}

} // namespace

void Mdl_compiled_material::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    tmp = m_vf->dump( transaction, m_arguments.get(), /*name*/ nullptr);
    s << "Arguments: " << tmp->get_c_str() << std::endl;

    tmp = m_ef->dump( transaction, m_temporaries.get(), /*name*/ nullptr);
    s << "Temporaries: " << tmp->get_c_str() << std::endl;

    tmp = m_ef->dump( transaction, m_body.get(), /*name*/ nullptr);
    s << "Body: " << tmp->get_c_str() << std::endl;

    char buffer[36];
    snprintf( buffer, sizeof( buffer),
        "%08x %08x %08x %08x", m_hash.m_id1, m_hash.m_id2, m_hash.m_id3, m_hash.m_id4);
    s << "Hash: " << buffer << std::endl;

    for( mi::Size i = 0; i < mi::mdl::IGenerated_code_dag::IMaterial_instance::MS_LAST+1; ++i) {
        const mi::base::Uuid& hash = m_slot_hashes[i];
        snprintf( buffer, sizeof( buffer),
            "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
        s << "Slot hash[" << i << "]: " << buffer << std::endl;

    }
    s << "Meters per scene unit: " << m_mdl_meters_per_scene_unit << std::endl;
    s << "Wavelength min: " << m_mdl_wavelength_min << std::endl;
    s << "Wavelength max: " << m_mdl_wavelength_max << std::endl;
    s << "Properties: " << m_properties << std::endl;
    s << "Internal space: " << m_internal_space << std::endl;

    s << "Opacity: " << get_opacity_str( m_opacity) << std::endl;
    s << "Surface opacity: " << get_opacity_str( m_surface_opacity) << std::endl;

    if( m_has_cutout_opacity)
        s << "Cutout_opacity: " << m_cutout_opacity << std::endl;
    else
        s << "Cutout_opacity: <Unknown>" << std::endl;

    s << "Resource table size: " <<  m_resources.size() << std::endl;
    for( mi::Size i = 0, n = m_resources.size(); i < n; ++i) {
        const Resource_tag_tuple& tuple = m_resources[i];
        s << "Resource " << i << ": kind " << tuple.m_kind
          << ",  MDL file path \"" << tuple.m_mdl_file_path
          << "\", selector \"" << tuple.m_selector
          << "\", tag " << tuple.m_tag.get_uint() << std::endl;
    }

    s << std::endl;
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

const SERIAL::Serializable* Mdl_compiled_material::serialize(
    SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    m_ef->serialize( serializer, m_body.get());
    m_ef->serialize_list( serializer, m_temporaries.get());
    m_vf->serialize_list( serializer, m_arguments.get());
    SERIAL::write( serializer, m_resources);

    write( serializer, m_hash);
    for( int i = 0; i < mi::mdl::IGenerated_code_dag::IMaterial_instance::MS_LAST+1; ++i)
        write( serializer, m_slot_hashes[i]);

    SERIAL::write( serializer, m_mdl_meters_per_scene_unit);
    SERIAL::write( serializer, m_mdl_wavelength_min);
    SERIAL::write( serializer, m_mdl_wavelength_max);
    SERIAL::write( serializer, m_properties);

    SERIAL::write( serializer, m_referenced_scene_data);

    SERIAL::write( serializer, m_internal_space);
    SERIAL::write_enum( serializer, m_opacity);
    SERIAL::write_enum( serializer, m_surface_opacity);
    SERIAL::write( serializer, m_cutout_opacity);
    SERIAL::write( serializer, m_has_cutout_opacity);
    SERIAL::write( serializer, m_module_idents);
    return this + 1;
}

SERIAL::Serializable* Mdl_compiled_material::deserialize(
    SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    mi::base::Handle<IExpression> body( m_ef->deserialize( deserializer));
    m_body        = body->get_interface<IExpression_direct_call>();
    m_temporaries = m_ef->deserialize_list( deserializer);
    m_arguments   = m_vf->deserialize_list( deserializer);

    SERIAL::read(deserializer, &m_resources);

    read( deserializer, &m_hash);
    for( int i = 0; i < mi::mdl::IGenerated_code_dag::IMaterial_instance::MS_LAST+1; ++i)
        read( deserializer, &m_slot_hashes[i]);

    SERIAL::read( deserializer, &m_mdl_meters_per_scene_unit);
    SERIAL::read( deserializer, &m_mdl_wavelength_min);
    SERIAL::read( deserializer, &m_mdl_wavelength_max);
    SERIAL::read( deserializer, &m_properties);

    SERIAL::read( deserializer, &m_referenced_scene_data);

    SERIAL::read( deserializer, &m_internal_space);

    SERIAL::read_enum( deserializer, &m_opacity);
    SERIAL::read_enum( deserializer, &m_surface_opacity);
    SERIAL::read( deserializer, &m_cutout_opacity);
    SERIAL::read( deserializer, &m_has_cutout_opacity);
    SERIAL::read( deserializer, &m_module_idents);

    return this + 1;
}

size_t Mdl_compiled_material::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_compiled_material, Mdl_compiled_material::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_compiled_material, Mdl_compiled_material::id>)
        + dynamic_memory_consumption( m_body)
        + dynamic_memory_consumption( m_temporaries)
        + dynamic_memory_consumption( m_arguments)
        + dynamic_memory_consumption( m_module_idents);
}

DB::Journal_type Mdl_compiled_material::get_journal_flags() const
{
    return DB::JOURNAL_NONE;
}

Uint Mdl_compiled_material::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_compiled_material::get_scene_element_references( DB::Tag_set* result) const
{
    collect_references( m_body.get(), result);
    collect_references( m_temporaries.get(), result);
    collect_references( m_arguments.get(), result);

    for( const auto& res: m_resources)
        if( res.m_tag)
            result->insert( res.m_tag);

    for( const auto& mod: m_module_idents)
        result->insert( mod.first);
}

} // namespace MDL

} // namespace MI
