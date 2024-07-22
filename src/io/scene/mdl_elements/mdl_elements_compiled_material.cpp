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

#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/tokenizer.hpp>

#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_module.h"
#include "mdl_elements_utilities.h"

#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_printers.h>
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

namespace MI {

namespace MDL {

Mdl_compiled_material::Mdl_compiled_material()
  : m_tf( get_type_factory()),
    m_vf( get_value_factory()),
    m_ef( get_expression_factory())
{
}

Mdl_compiled_material::Mdl_compiled_material(
    DB::Transaction* transaction,
    const mi::mdl::IMaterial_instance* core_material_instance,
    const char* module_name,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max,
    bool resolve_resources)
  : m_tf( get_type_factory()),
    m_vf( get_value_factory()),
    m_ef( get_expression_factory()),
    m_core_material_instance( core_material_instance, mi::base::DUP_INTERFACE),
    m_mdl_meters_per_scene_unit( mdl_meters_per_scene_unit),
    m_mdl_wavelength_min( mdl_wavelength_min),
    m_mdl_wavelength_max( mdl_wavelength_max),
    m_resolve_resources( resolve_resources)
{
    ASSERT( M_SCENE, core_material_instance);

#if 0
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IOutput_stream> os_stdout(
        mdl->create_std_stream( mi::mdl::IMDL::OS_STDOUT));
    mi::base::Handle<mi::mdl::IPrinter> printer( mdl->create_printer( os_stdout.get()));
    printer->print( core_material_instance);
#endif

    // Collect all referenced tags and identifiers of user modules in body, temporaries and
    // parameters.

    Mdl_dag_converter_light converter(
        transaction,
        core_material_instance->get_resource_tagger(),
        &m_tags,
        &m_module_idents);

    const mi::mdl::DAG_call* constructor = core_material_instance->get_constructor();
    converter.process_dag_node( constructor);

    mi::Size n = core_material_instance->get_temporary_count();
    for( mi::Size i = 0; i < n; ++i) {
        const mi::mdl::DAG_node* temporary = core_material_instance->get_temporary_value( i);
        converter.process_dag_node( temporary);
    }

    n = core_material_instance->get_parameter_count();
    for( mi::Size i = 0; i < n; ++i) {
        const mi::mdl::IValue* argument = core_material_instance->get_parameter_default( i);
        converter.process_value( argument);
    }

    // Collect module identifier from originating module.
    if( module_name) {
        DB::Tag module_tag = transaction->name_to_tag( get_db_name( module_name).c_str());
        DB::Access<Mdl_module> module( module_tag, transaction);
        m_module_idents.insert( {module_tag, module->get_ident()});
    }

    // Copy the resource tag table.
    n = core_material_instance->get_resource_tag_map_entries_count();
    for( mi::Size i = 0; i < n; ++i) {
        const mi::mdl::Resource_tag_tuple* e
            = core_material_instance->get_resource_tag_map_entry( i);
        m_resources.emplace_back( *e);
    }
}

const IExpression_direct_call* Mdl_compiled_material::get_body( DB::Transaction* transaction) const
{
    std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));
    const mi::mdl::DAG_call* constructor = m_core_material_instance->get_constructor();
    mi::base::Handle<const IExpression> body(
        converter->core_dag_node_to_int_expr( constructor, /*type_int*/ nullptr));
    return body->get_interface<IExpression_direct_call>();
}

mi::Size Mdl_compiled_material::get_temporary_count() const
{
    return m_core_material_instance->get_temporary_count();
}

const IExpression* Mdl_compiled_material::get_temporary(
    DB::Transaction* transaction, mi::Size index) const
{
    if( index >= get_temporary_count())
        return nullptr;

    std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));
    const mi::mdl::DAG_node* temporary = m_core_material_instance->get_temporary_value( index);
    return converter->core_dag_node_to_int_expr( temporary, /*type_int*/ nullptr);
}

namespace {

/// Converts path from the SDK representation (dots and array index brackets) to the MDL core
/// representation (dots only).
///
/// The simple search-and-replace fails to reject some invalid input like "foo[bar".
std::string convert_path( const char* path)
{
    std::string result = path;
    boost::replace_all( result, "[", ".");
    boost::erase_all( result, "]");
    return result;
}

} // namespace

const IExpression* Mdl_compiled_material::lookup_sub_expression(
    DB::Transaction* transaction, const char* path) const
{
    ASSERT( M_SCENE, path);

    std::string core_path = convert_path( path);

    const mi::mdl::DAG_node* sub_expr_dag_node;
    const mi::mdl::IValue* sub_expr_value;
    m_core_material_instance->lookup_sub_expression(
        core_path.c_str(), sub_expr_dag_node, sub_expr_value);

    if( sub_expr_dag_node) {
         std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));
         return converter->core_dag_node_to_int_expr( sub_expr_dag_node, /*type_int*/ nullptr);
    }

    if( sub_expr_value) {
         std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));
         mi::base::Handle<IValue> value(
             converter->core_value_to_int_value( /*type_int*/ nullptr, sub_expr_value));
         ASSERT( M_SCENE, value);
         return m_ef->create_constant( value.get());
    }

    return nullptr;
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

mi::Size Mdl_compiled_material::get_parameter_count() const
{
    return m_core_material_instance->get_parameter_count();
}

const char* Mdl_compiled_material::get_parameter_name( mi::Size index) const
{
    if( index >= get_parameter_count())
        return nullptr;

    const char* name = m_core_material_instance->get_parameter_name( index);
    ASSERT( M_SCENE, name);
    return name;
}

const IValue* Mdl_compiled_material::get_argument(
    DB::Transaction* transaction, mi::Size index) const
{
    if( index >= get_parameter_count())
        return nullptr;

    std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));
    const mi::mdl::IValue* argument = m_core_material_instance->get_parameter_default( index);
    return converter->core_value_to_int_value( /*type_int*/ nullptr, argument);
}

namespace {

DB::Tag get_next_call(const Mdl_function_call* fc, const std::string& parameter_name)
{
    mi::Size p_index = fc->get_parameter_index(parameter_name.c_str());
    if (p_index == static_cast<mi::Size>(-1))
        return {};

    mi::base::Handle<const IExpression_list> arguments(fc->get_arguments());
    mi::base::Handle<const IExpression> expr(arguments->get_expression(p_index));
    mi::base::Handle<const IExpression_call> expr_call(
        expr->get_interface<const IExpression_call>());
    if (!expr_call)
        return {};

    return expr_call->get_call();
}

} // namespace

DB::Tag Mdl_compiled_material::get_connected_function_db_name(
    DB::Transaction* transaction,
    DB::Tag material_instance_tag,
    mi::Size parameter_index) const
{
    DB::Access<Mdl_function_call> material_instance( material_instance_tag, transaction);

    std::vector<std::string> path_tokens;
    std::string parameter_name = get_parameter_name( parameter_index);
    boost::split( path_tokens, parameter_name, boost::is_any_of("."));

    // There needs to be at least one item, last one is the param name.
    // For struct constructors attached to the material, there is only one token.
    // For other attached functions there are more.
    if (path_tokens.size() == 0)
        return {};

    DB::Tag call_tag = material_instance_tag;
    for( auto& path_token : path_tokens)
    {
        SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
        if (class_id != MDL::ID_MDL_FUNCTION_CALL) {
            ASSERT(M_SCENE, false);
            return {};
        }

        DB::Access<Mdl_function_call> fc(call_tag, transaction);
        DB::Tag next_tag = get_next_call(fc.get_ptr(), path_token);
        if (!next_tag)
            break;

        call_tag = next_tag;
        continue;
    }

    // if no function is attached, the zero-tag is returned
    return call_tag == material_instance_tag ? DB::Tag() : call_tag;
}

mi::mdl::IMaterial_instance::Opacity Mdl_compiled_material::get_opacity() const
{
    return m_core_material_instance->get_opacity();
}

mi::mdl::IMaterial_instance::Opacity Mdl_compiled_material::get_surface_opacity() const
{
    return m_core_material_instance->get_surface_opacity();
}

bool Mdl_compiled_material::get_cutout_opacity( mi::Float32* cutout_opacity) const
{
    const mi::mdl::IValue_float* value = m_core_material_instance->get_cutout_opacity();
    if( !value)
        return false;

    if( cutout_opacity)
        *cutout_opacity = value->get_value();
    return true;
}

mi::Size Mdl_compiled_material::get_referenced_scene_data_count() const
{
    return m_core_material_instance->get_referenced_scene_data_count();
}

const char* Mdl_compiled_material::get_referenced_scene_data_name( mi::Size index) const
{
    return m_core_material_instance->get_referenced_scene_data_name( index);
}

bool Mdl_compiled_material::depends_on_state_transform() const
{
    auto properties = m_core_material_instance->get_properties();
    return (properties & mi::mdl::IMaterial_instance::IP_DEPENDS_ON_TRANSFORM) != 0;
}

bool Mdl_compiled_material::depends_on_state_object_id() const
{
    auto properties = m_core_material_instance->get_properties();
    return (properties & mi::mdl::IMaterial_instance::IP_DEPENDS_ON_OBJECT_ID) != 0;
}

bool Mdl_compiled_material::depends_on_global_distribution() const
{
    auto properties = m_core_material_instance->get_properties();
    return (properties & mi::mdl::IMaterial_instance::IP_DEPENDS_ON_GLOBAL_DISTRIBUTION) != 0;
}

bool Mdl_compiled_material::depends_on_uniform_scene_data() const
{
    auto properties = m_core_material_instance->get_properties();
    return (properties & mi::mdl::IMaterial_instance::IP_DEPENDS_ON_UNIFORM_SCENE_DATA) != 0;
}

mi::base::Uuid Mdl_compiled_material::get_hash() const
{
     const mi::mdl::DAG_hash* hash = m_core_material_instance->get_hash();
     return convert_hash( *hash);
}

mi::base::Uuid Mdl_compiled_material::get_slot_hash( mi::neuraylib::Material_slot slot) const
{
    mi::mdl::IMaterial_instance::Slot core_slot = ext_slot_to_core_lost( slot);
    const mi::mdl::DAG_hash* hash = m_core_material_instance->get_slot_hash( core_slot);
    return convert_hash( *hash);
}

mi::base::Uuid  Mdl_compiled_material::get_sub_expression_hash( const char* path) const
{
    ASSERT( M_SCENE, path);

    std::string core_path = convert_path( path);
    mi::mdl::DAG_hash hash = m_core_material_instance->get_sub_expression_hash( core_path.c_str());
    return convert_hash( hash);
}

const mi::mdl::IMaterial_instance* Mdl_compiled_material::get_core_material_instance() const
{
    m_core_material_instance->retain();
    return m_core_material_instance.get();
}

const char* Mdl_compiled_material::get_internal_space() const
{
    return m_core_material_instance->get_internal_space();
}

mi::Size Mdl_compiled_material::get_resources_count() const
{
    return m_resources.size();
}

const Resource_tag_tuple* Mdl_compiled_material::get_resource_tag_tuple( mi::Size index) const
{
    if( index >= m_resources.size())
        return nullptr;

    return &m_resources[index];
}

const IValue_list* Mdl_compiled_material::get_arguments( DB::Transaction* transaction) const
{
    std::unique_ptr<Mdl_dag_converter> converter( get_dag_converter( transaction));

    mi::Size n = m_core_material_instance->get_parameter_count();
    mi::base::Handle<IValue_list> arguments( m_vf->create_value_list( n));

    for( mi::Size i = 0; i < n; ++i) {
        const char* name = m_core_material_instance->get_parameter_name( i);
        const mi::mdl::IValue* core_argument = m_core_material_instance->get_parameter_default( i);
        mi::base::Handle<const IValue> argument(
            converter->core_value_to_int_value( /*type_int*/ nullptr, core_argument));
        ASSERT( M_SCENE, argument);
        arguments->add_value_unchecked( name, argument.get());
    }

    return arguments.extract();
}

void Mdl_compiled_material::swap( Mdl_compiled_material& other)
{
    SCENE::Scene_element<Mdl_compiled_material, ID_MDL_COMPILED_MATERIAL>::swap( other);

    std::swap( m_core_material_instance, other.m_core_material_instance);
    m_resources.swap( other.m_resources);

    std::swap( m_mdl_meters_per_scene_unit, other.m_mdl_meters_per_scene_unit);
    std::swap( m_mdl_wavelength_min, other.m_mdl_wavelength_min);
    std::swap( m_mdl_wavelength_max, other.m_mdl_wavelength_max);
    std::swap( m_resolve_resources, other.m_resolve_resources);

    std::swap( m_tags, other.m_tags);
    std::swap( m_module_idents, other.m_module_idents);
}

const SERIAL::Serializable* Mdl_compiled_material::serialize(
    SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mdlc_module->serialize_material_instance( serializer, m_core_material_instance.get());

    SERIAL::write( serializer, m_resources);

    SERIAL::write( serializer, m_mdl_meters_per_scene_unit);
    SERIAL::write( serializer, m_mdl_wavelength_min);
    SERIAL::write( serializer, m_mdl_wavelength_max);
    SERIAL::write( serializer, m_resolve_resources);

    SERIAL::write( serializer, m_tags);
    SERIAL::write( serializer, m_module_idents);
    return this + 1;
}

SERIAL::Serializable* Mdl_compiled_material::deserialize(
    SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    m_core_material_instance = mdlc_module->deserialize_material_instance( deserializer);

    SERIAL::read( deserializer, &m_resources);

    SERIAL::read( deserializer, &m_mdl_meters_per_scene_unit);
    SERIAL::read( deserializer, &m_mdl_wavelength_min);
    SERIAL::read( deserializer, &m_mdl_wavelength_max);
    SERIAL::read( deserializer, &m_resolve_resources);

    SERIAL::read( deserializer, &m_tags);
    SERIAL::read( deserializer, &m_module_idents);
    return this + 1;
}

void Mdl_compiled_material::dump() const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    s << "Meters per scene unit: " << m_mdl_meters_per_scene_unit << std::endl;
    s << "Wavelength min: " << m_mdl_wavelength_min << std::endl;
    s << "Wavelength max: " << m_mdl_wavelength_max << std::endl;

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

size_t Mdl_compiled_material::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_compiled_material, ID_MDL_COMPILED_MATERIAL>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_compiled_material, ID_MDL_COMPILED_MATERIAL>)
        + m_core_material_instance->get_memory_size()
        + dynamic_memory_consumption( m_tags)
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
    for( const auto& resource: m_resources)
        if( resource.m_tag)
            result->insert( resource.m_tag);

    result->insert( m_tags.begin(), m_tags.end());

    for( const auto& module_ident: m_module_idents)
        result->insert( module_ident.first);
}

std::unique_ptr<Mdl_dag_converter> Mdl_compiled_material::get_dag_converter(
    DB::Transaction* transaction) const
{
    // Do *not* pass module_name to the DAG converter. This is only necessary for localization of
    // annnotations, which does not matter here. Avoids misuse of this information for other (wrong)
    // purposes.
    return std::make_unique<Mdl_dag_converter>(
        m_ef.get(),
        transaction,
        m_core_material_instance->get_resource_tagger(),
        /*code_dag*/ nullptr,
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_name*/ nullptr,
        /*prototype_tag*/ DB::Tag(),
        m_resolve_resources,
        /*user_modules_seen*/ nullptr);
}

} // namespace MDL

} // namespace MI
