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

#include "pch.h"

#include "i_mdl_elements_material_definition.h"

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_material_instance.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_utilities.h"

#include <sstream>
#include <mi/neuraylib/istring.h>
#include <mi/mdl/mdl_archiver.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/scene/i_scene_journal_types.h>


namespace MI {

namespace MDL {

Mdl_material_definition::Mdl_material_definition()
: m_tf(get_type_factory())
, m_vf(get_value_factory())
, m_ef(get_expression_factory())
, m_module_tag()
, m_material_tag()
, m_material_index(~0u)
, m_name()
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported(false)
, m_call_references()
, m_parameter_types()
, m_defaults()
, m_annotations()
, m_parameter_annotations()
, m_enable_if_conditions()
, m_enable_if_users()
{
}

Mdl_material_definition::Mdl_material_definition(
    DB::Transaction* transaction,
    DB::Tag module_tag,
    DB::Tag material_tag,
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Uint32 material_index,
    const char* module_filename,
    const char* module_name)
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_tag( module_tag)
, m_material_tag( material_tag)
, m_material_index( material_index)
, m_name( code_dag->get_material_name( material_index))
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported( code_dag->get_material_exported( material_index))
, m_call_references()
, m_parameter_types()
, m_defaults()
, m_annotations()
, m_parameter_annotations()
, m_enable_if_conditions()
, m_enable_if_users()
{
    const char* s = code_dag->get_cloned_material_name( material_index);
    std::string prototype_name = s == NULL ? "" : s;
    m_prototype_tag = prototype_name.empty()
        ? DB::Tag() : transaction->name_to_tag( add_mdl_db_prefix( prototype_name).c_str());
    ASSERT( M_SCENE, m_prototype_tag || prototype_name.empty());

    const char* original_name = code_dag->get_original_material_name( material_index);
    if( original_name)
        m_original_name = original_name;

    m_tf = get_type_factory();
    m_vf = get_value_factory();
    m_ef = get_expression_factory();

    // material annotations
    mi::Uint32 annotation_count = code_dag->get_material_annotation_count( material_index);
    Mdl_annotation_block annotations( annotation_count);
    for( mi::Uint32 i = 0; i < annotation_count; ++i)
        annotations[i] = code_dag->get_material_annotation( material_index, i);
    m_annotations = mdl_dag_node_vector_to_int_annotation_block(
        m_ef.get(), transaction, annotations, module_filename, module_name);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    MDLC::Mdlc_module::Register_mdl_type_with_api* register_type_callback
        = mdlc_module->get_register_mdl_type_with_api_callback();

    // parameters/arguments
    m_defaults = m_ef->create_expression_list();
    m_parameter_annotations = m_ef->create_annotation_list();
    m_parameter_types = m_tf->create_type_list();
    m_enable_if_conditions = m_ef->create_expression_list();

    mi::Uint32 parameter_count = code_dag->get_material_parameter_count( material_index);
    m_enable_if_users.resize( parameter_count);

    for( mi::Uint32 i = 0; i < parameter_count; ++i) {

        const char* parameter_name = code_dag->get_material_parameter_name( material_index, i);

        // update m_parameter_types
        const mi::mdl::IType* parameter_type
            = code_dag->get_material_parameter_type( material_index, i);
        mi::base::Handle<const IType> type( mdl_type_to_int_type( m_tf.get(), parameter_type));
        m_parameter_types->add_type( parameter_name, type.get());
        if( register_type_callback)
            register_type_callback( type.get());

        // update m_defaults
        const mi::mdl::DAG_node* default_
            = code_dag->get_material_parameter_default( material_index, i);
        if( default_) {
            mi::base::Handle<IExpression> default_int( mdl_dag_node_to_int_expr(
                m_ef.get(),
                transaction,
                type.get(),
                default_,
                /*immutable*/ true,
                /*create_direct_calls*/ false,
                module_filename,
                module_name));
            ASSERT( M_SCENE, default_int);
            m_defaults->add_expression( parameter_name, default_int.get());
        }

        // update enable_if conditions
        const mi::mdl::DAG_node* enable_if_cond
            = code_dag->get_material_parameter_enable_if_condition( material_index, i);
        if (enable_if_cond) {
            mi::base::Handle<IExpression> enable_if_cond_int(mdl_dag_node_to_int_expr(
                m_ef.get(),
                transaction,
                type.get(),
                enable_if_cond,
                /*immutable*/ true,
                /*create_direct_calls*/ false,
                module_filename,
                module_name));
            ASSERT(M_SCENE, enable_if_cond_int);
            m_enable_if_conditions->add_expression(parameter_name, enable_if_cond_int.get());
        }
        std::vector<mi::Sint32> &users = m_enable_if_users[i];
        mi::Size n_users = code_dag->get_material_parameter_enable_if_condition_users(
            material_index, i);
        for (size_t j = 0; j < n_users; ++j) {
            int param_idx = code_dag->get_material_parameter_enable_if_condition_user(
                material_index, i, int(j));
            users.push_back(param_idx);
        }

        // update m_parameter_annotations
        mi::Uint32 parameter_annotation_count
            = code_dag->get_material_parameter_annotation_count( material_index, i);
        Mdl_annotation_block parameter_annotations( parameter_annotation_count);
        for( mi::Uint32 j = 0; j < parameter_annotation_count; ++j)
            parameter_annotations[j]
                = code_dag->get_material_parameter_annotation( material_index, i, j);
        mi::base::Handle<IAnnotation_block> block( mdl_dag_node_vector_to_int_annotation_block(
            m_ef.get(), transaction, parameter_annotations, module_filename, module_name));
        if( block)
            m_parameter_annotations->add_annotation_block( parameter_name, block.get());
    }

    // update m_call_references
    collect_material_references( transaction, code_dag, material_index, m_call_references);
    
    if (m_is_exported && module_filename)
    {
        mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());
        m_thumbnail = DETAIL::lookup_thumbnail(
            module_filename, m_name, m_annotations.get(), archive_tool.get());
    }
}

DB::Tag Mdl_material_definition::get_module() const
{
    return m_module_tag;
}

const char* Mdl_material_definition::get_mdl_name() const
{
    return m_name.c_str();
}

DB::Tag Mdl_material_definition::get_prototype() const
{
    return m_prototype_tag;
}

bool Mdl_material_definition::is_exported() const
{
    return m_is_exported;
}

mi::Size Mdl_material_definition::get_parameter_count() const
{
    return m_parameter_types->get_size();
}

const char* Mdl_material_definition::get_parameter_name( mi::Size index) const
{
    return m_parameter_types->get_name( index);
}

mi::Size Mdl_material_definition::get_parameter_index( const char* name) const
{
    return m_parameter_types->get_index( name);
}

const IType_list* Mdl_material_definition::get_parameter_types() const
{
    m_parameter_types->retain();
    return m_parameter_types.get();
}

const IExpression_list* Mdl_material_definition::get_defaults() const
{
    m_defaults->retain();
    return m_defaults.get();
}

const IExpression_list* Mdl_material_definition::get_enable_if_conditions() const
{
    m_enable_if_conditions->retain();
    return m_enable_if_conditions.get();
}

mi::Size Mdl_material_definition::get_enable_if_users( mi::Size index) const
{
    if (index < m_enable_if_users.size())
        return m_enable_if_users[index].size();
    return 0;
}

mi::Size Mdl_material_definition::get_enable_if_user(mi::Size index, mi::Size u_index) const
{
    if (index < m_enable_if_users.size()) {
        if (u_index < m_enable_if_users[index].size()) {
            return m_enable_if_users[index][u_index];
        }
    }
    return ~mi::Size(0);
}

const IAnnotation_block* Mdl_material_definition::get_annotations() const
{
    if( !m_annotations)
        return 0;
    m_annotations->retain();
    return m_annotations.get();
}

const IAnnotation_list* Mdl_material_definition::get_parameter_annotations() const
{
    m_parameter_annotations->retain();
    return m_parameter_annotations.get();
}

const char* Mdl_material_definition::get_thumbnail() const
{
    return m_thumbnail.empty() ? 0 : m_thumbnail.c_str();
}

Mdl_material_instance* Mdl_material_definition::create_material_instance(
    DB::Transaction* transaction, const IExpression_list* arguments, mi::Sint32* errors) const
{
    return create_material_instance_internal(
        transaction, arguments, /*allow_ek_parameter*/ false, /*immutable*/ false, errors);
}

Mdl_material_instance* Mdl_material_definition::create_material_instance_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool allow_ek_parameter,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (errors == NULL)
        errors = &dummy_errors;

    // prevent instantiation of non-exported materials
    if( !m_is_exported) {
        *errors = -4;
        return NULL;
    }

    // check that the provided arguments are parameters of the material definition and that their
    // types match the expected types
    if( arguments) {
        mi::Size n = arguments->get_size();
        for( mi::Size i = 0; i < n; ++i) {
            const char* name = arguments->get_name( i);
            mi::base::Handle<const IType> expected_type( m_parameter_types->get_type( name));
            if( !expected_type) {
                *errors = -1;
                return NULL;
            }
            mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
            mi::base::Handle<const IType> actual_type( argument->get_type());
            if( !argument_type_matches_parameter_type(
                m_tf.get(), actual_type.get(), expected_type.get())) {
                *errors = -2;
                return NULL;
            }
            bool actual_type_varying
                = (actual_type->get_all_type_modifiers()   & IType::MK_VARYING) != 0;
            bool expected_type_uniform
                = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
            if( actual_type_varying && expected_type_uniform) {
                *errors = -5;
                return NULL;
            }
            IExpression::Kind kind = argument->get_kind();
            if(     kind != IExpression::EK_CONSTANT
                &&  kind != IExpression::EK_CALL
                && (kind != IExpression::EK_PARAMETER || !allow_ek_parameter)) {
                *errors = -6;
                return NULL;
            }
            if( expected_type_uniform && return_type_is_varying( transaction, argument.get())) {
                *errors = -8;
                return NULL;
            }
        }
    }

    // build up complete argument set using the defaults where necessary
    mi::base::Handle<IExpression_list> complete_arguments( m_ef->create_expression_list());
    std::vector<mi::base::Handle<const IExpression> > context;
    for( mi::Size i = 0, n = m_parameter_types->get_size(); i < n; ++i) {
        const char* name = get_parameter_name( i);
        mi::base::Handle<const IExpression> argument(
            arguments ? arguments->get_expression( name) : NULL);
        if( argument) {
            // use provided argument
            mi::base::Handle<IExpression> argument_copy( m_ef->clone( argument.get()));
            ASSERT( M_SCENE, argument_copy);
            argument = argument_copy;
        } else {
            // no argument provided, use default
            mi::base::Handle<const IExpression> default_( m_defaults->get_expression( name));
            if( !default_) {
                // no default available
                *errors = -3;
                return NULL;
            }
            mi::base::Handle<const IType> expected_type( m_parameter_types->get_type( name));
            bool expected_type_uniform
                = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
            if( expected_type_uniform && return_type_is_varying( transaction, default_.get())) {
                *errors = -8;
                return NULL;
            }
            // use clone of default as argument
            mi::base::Handle<IExpression> default_copy(
                deep_copy( m_ef.get(), transaction, default_.get(), context));
            ASSERT( M_SCENE, default_copy);
            argument = default_copy;
        }
        complete_arguments->add_expression( name, argument.get());
        context.push_back( argument);
    }

    Mdl_material_instance* instance = new Mdl_material_instance(
        m_material_tag, m_material_index, complete_arguments.get(), m_name.c_str(),
        m_parameter_types.get(),
        immutable,
        m_enable_if_conditions.get());
    *errors = 0;
    return instance;
}

const mi::mdl::IType* Mdl_material_definition::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    DB::Access<Mdl_module> module( m_module_tag, transaction);
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_material_parameter_type( m_material_index, index);
}

const char* Mdl_material_definition::get_mdl_original_name() const
{
    return m_original_name.empty() ? 0 : m_original_name.c_str();
}

const SERIAL::Serializable* Mdl_material_definition::serialize( SERIAL::Serializer* serializer)const
{
    Scene_element_base::serialize( serializer);

    serializer->write( m_module_tag);
    serializer->write( m_material_tag);
    serializer->write( m_material_index);
    serializer->write( m_name);
    serializer->write( m_original_name);
    serializer->write( m_thumbnail);
    serializer->write( m_prototype_tag);
    serializer->write( m_is_exported);
    SERIAL::write( serializer, m_call_references);

    m_tf->serialize_list( serializer, m_parameter_types.get());
    m_ef->serialize_list( serializer, m_defaults.get());
    m_ef->serialize_annotation_block( serializer, m_annotations.get());
    m_ef->serialize_annotation_list( serializer, m_parameter_annotations.get());
    m_ef->serialize_list( serializer, m_enable_if_conditions.get());

    serializer->write( m_enable_if_users);

    return this + 1;
}

SERIAL::Serializable* Mdl_material_definition::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_module_tag);
    deserializer->read( &m_material_tag);
    deserializer->read( &m_material_index);
    deserializer->read( &m_name);
    deserializer->read( &m_original_name);
    deserializer->read( &m_thumbnail);
    deserializer->read( &m_prototype_tag);
    deserializer->read( &m_is_exported);
    SERIAL::read( deserializer, &m_call_references);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    m_defaults = m_ef->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_parameter_annotations = m_ef->deserialize_annotation_list( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    deserializer->read( &m_enable_if_users);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    MDLC::Mdlc_module::Register_mdl_type_with_api* register_type_callback
        = mdlc_module->get_register_mdl_type_with_api_callback();
    if( register_type_callback) {
        mi::Size n = m_parameter_types->get_size();
        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<const IType> type( m_parameter_types->get_type( i));
            register_type_callback( type.get());
        }
    }

    if( !m_thumbnail.empty())
    {
        m_thumbnail = HAL::Ospath::convert_to_platform_specific_path( m_thumbnail);
        if( !DISK::access( m_thumbnail.c_str()))
            m_thumbnail.clear();
    }
    return this + 1;
}

void Mdl_material_definition::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    s << "Module tag: " << m_module_tag.get_uint() << std::endl;
    s << "Material definition tag: " << m_material_tag.get_uint() << std::endl;
    s << "Material index: " << m_material_index << std::endl;
    s << "Material definition MDL name: " << m_name << std::endl;
    s << "Material definition MDL original name: " << m_original_name << std::endl;
    s << "Prototype tag: " << m_prototype_tag.get_uint() << std::endl;
    s << "Is exported: " << m_is_exported << std::endl;
    // m_call_references missing
    tmp = m_tf->dump( m_parameter_types.get());
    s << "Parameter types: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_defaults.get(), /*name*/ 0);
    s << "Defaults: " << tmp->get_c_str() << std::endl;
    s << "Annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_parameter_annotations.get(), /*name*/ 0);
    s << "Parameter annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_enable_if_conditions.get(), /*name*/ 0);
    s << "Enable_if conditions: " << tmp->get_c_str() << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_material_definition::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_material_definition, Mdl_material_definition::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_material_definition, Mdl_material_definition::id>)
        + dynamic_memory_consumption( m_name)
        + dynamic_memory_consumption( m_original_name)
        + dynamic_memory_consumption( m_thumbnail)
        + dynamic_memory_consumption( m_parameter_types)
        + dynamic_memory_consumption( m_call_references)
        + dynamic_memory_consumption( m_defaults)
        + dynamic_memory_consumption( m_annotations)
        + dynamic_memory_consumption( m_parameter_annotations)
        + dynamic_memory_consumption( m_enable_if_conditions);
}

DB::Journal_type Mdl_material_definition::get_journal_flags() const
{
    return DB::JOURNAL_NONE;
}

Uint Mdl_material_definition::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_material_definition::get_scene_element_references( DB::Tag_set* result) const
{
    result->insert( m_module_tag);
    // skip m_material_tag (own tag)
    if( m_prototype_tag)
        result->insert( m_prototype_tag);
    result->insert( m_call_references.begin(), m_call_references.end());
    collect_references( m_defaults.get(), result);
    collect_references( m_annotations.get(), result);
    collect_references( m_parameter_annotations.get(), result);
    collect_references( m_enable_if_conditions.get(), result);
}

} // namespace MDL

} // namespace MI
