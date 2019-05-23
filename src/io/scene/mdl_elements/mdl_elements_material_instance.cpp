/***************************************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_material_instance.h"

#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_material_definition.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_utilities.h"

#include <sstream>
#include <mi/base/handle.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_access.h>
#include <base/data/serial/i_serializer.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

namespace MI {

namespace MDL {

Mdl_material_instance::Mdl_material_instance()
: m_tf(get_type_factory())
, m_vf(get_value_factory())
, m_ef(get_expression_factory())
, m_module_tag()
, m_definition_tag()
, m_material_index( ~0u)
, m_definition_name()
, m_immutable( false) // avoid ubsan warning with swap() and temporaries
, m_parameter_types()
, m_arguments()
, m_enable_if_conditions()
{
}

static inline char const *check_str(char const *s)
{
    ASSERT(M_SCENE, s != NULL && "string argument should be non-NULL");
    return s;
}

Mdl_material_instance::Mdl_material_instance(
    DB::Tag module_tag,
    DB::Tag definition_tag,
    mi::Uint32 material_index,
    IExpression_list* arguments,
    const char* definition_name,
    const IType_list* parameter_types,
    bool immutable,
    const IExpression_list* enable_if_conditions)
: m_tf(get_type_factory())
, m_vf(get_value_factory())
, m_ef(get_expression_factory())
, m_module_tag(module_tag)
, m_definition_tag(definition_tag)
, m_material_index(material_index)
, m_definition_name(check_str(definition_name))
, m_immutable(immutable)
, m_parameter_types(make_handle_dup(parameter_types))
, m_arguments(make_handle_dup(arguments))
, m_enable_if_conditions(make_handle_dup(enable_if_conditions))
{
}

Mdl_material_instance::Mdl_material_instance( const Mdl_material_instance& other)
: SCENE::Scene_element<Mdl_material_instance, ID_MDL_MATERIAL_INSTANCE>( other)
, m_tf( other.m_tf)
, m_vf( other.m_vf)
, m_ef( other.m_ef)
, m_module_tag(other.m_module_tag)
, m_definition_tag( other.m_definition_tag)
, m_material_index( other.m_material_index)
, m_definition_name( other.m_definition_name)
, m_immutable( other.m_immutable)
, m_parameter_types( other.m_parameter_types)
, m_arguments( m_ef->clone(
    other.m_arguments.get(), /*transaction*/ nullptr, /*copy_immutable_calls*/ false))
, m_enable_if_conditions( other.m_enable_if_conditions)  // shared, no clone necessary
{
}

DB::Tag Mdl_material_instance::get_material_definition() const
{
    ASSERT( M_SCENE, m_definition_tag.is_valid());
    return m_definition_tag;
}

const char* Mdl_material_instance::get_mdl_material_definition() const
{
    return m_definition_name.c_str();
}

mi::Size Mdl_material_instance::get_parameter_count() const
{
    return m_arguments->get_size();
}

const char* Mdl_material_instance::get_parameter_name( mi::Size index) const
{
    return m_arguments->get_name( index);
}

mi::Size Mdl_material_instance::get_parameter_index( const char* name) const
{
    return m_arguments->get_index( name);
}

const IType_list* Mdl_material_instance::get_parameter_types() const
{
    m_parameter_types->retain();
    return m_parameter_types.get();
}

const IExpression_list* Mdl_material_instance::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

// Get the list of enable_if conditions.
const IExpression_list* Mdl_material_instance::get_enable_if_conditions() const
{
    m_enable_if_conditions->retain();
    return m_enable_if_conditions.get();
}

mi::Sint32 Mdl_material_instance::set_arguments(
    DB::Transaction* transaction, const IExpression_list* arguments)
{
    if( !arguments)
        return -1;
    mi::Size n = arguments->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        const char* name = arguments->get_name( i);
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        mi::Sint32 result = set_argument( transaction, name, argument.get());
        if( result != 0)
            return result;
    }
    return 0;
}

mi::Sint32 Mdl_material_instance::set_argument(
    DB::Transaction* transaction, mi::Size index, const IExpression* argument)
{
    if (!argument)
        return -1;
    mi::base::Handle<const IType> expected_type(m_parameter_types->get_type( index));
    if (!expected_type)
        return -2;
    mi::base::Handle<const IType> actual_type(argument->get_type());

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();
    bool needs_cast = false;
    if (!argument_type_matches_parameter_type(
        m_tf.get(),
        actual_type.get(),
        expected_type.get(),
        allow_cast,
        needs_cast))
            return -3;

    if (m_immutable)
        return -4;

    bool actual_type_varying   = (actual_type->get_all_type_modifiers()   & IType::MK_VARYING) != 0;
    bool expected_type_uniform = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
    if (actual_type_varying && expected_type_uniform)
        return -5;

    IExpression::Kind kind = argument->get_kind();
    if (kind != IExpression::EK_CONSTANT && kind != IExpression::EK_CALL)
        return -6;

    if( expected_type_uniform && return_type_is_varying( transaction, argument))
        return -8;

    mi::base::Handle<IExpression> argument_copy(m_ef->clone(
        argument, transaction, /*copy_immutable_calls=*/ true));

    if (needs_cast) {
        mi::Sint32 errors = 0;
        argument_copy = m_ef->create_cast(
            transaction,
            argument_copy.get(),
            expected_type.get(),
            /*db_element_name=*/nullptr,
            /*force_cast=*/false,
            /*direct_call=*/false,
            &errors);
        ASSERT(M_SCENE, argument_copy); // should always succeed.
    }
   
    m_arguments->set_expression(index, argument_copy.get());
    return 0;
}

mi::Sint32 Mdl_material_instance::set_argument(
    DB::Transaction* transaction, const char* name, const IExpression* argument)
{
    if( !name || !argument)
        return -1;
    mi::Size index = get_parameter_index( name);
    return set_argument( transaction, index, argument);
}

void Mdl_material_instance::make_mutable(DB::Transaction* transaction) {

    // material instances, which are defaults in their own module, do not
    // keep a reference to their module, get it now
    if (!m_module_tag.is_valid()) {
        DB::Access<Mdl_material_definition> definition(m_definition_tag, transaction);
        m_module_tag = definition->get_module(transaction);
        ASSERT(M_SCENE, m_module_tag.is_valid());
    }
    m_immutable = false;
}

Mdl_compiled_material* Mdl_material_instance::create_compiled_material(
    DB::Transaction* transaction,
    bool class_compilation,
    Execution_context* context) const
{
    context->clear_messages();

    mi::base::Handle<const mi::mdl::IGenerated_code_dag::IMaterial_instance> instance(
        create_dag_material_instance( transaction, /*use_temporaries*/ true, class_compilation,
           context));
    if( !instance.is_valid_interface())
        return 0;

    ASSERT(M_SCENE, m_module_tag.is_valid());
    DB::Access<Mdl_material_definition> material_definition( m_definition_tag, transaction);
    DB::Access<Mdl_module> module( m_module_tag, transaction);
    const char* module_filename = module->get_filename();
    const char* module_name = module->get_mdl_name();

    mi::Float32 mdl_meters_per_scene_unit = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_METERS_PER_SCENE_UNIT);
    mi::Float32 mdl_wavelength_min = context->get_option<mi::Float32>(MDL_CTX_OPTION_WAVELENGTH_MIN);
    mi::Float32 mdl_wavelength_max = context->get_option<mi::Float32>(MDL_CTX_OPTION_WAVELENGTH_MAX);
    bool load_resources = context->get_option<bool>(MDL_CTX_OPTION_RESOLVE_RESOURCES);

    return new Mdl_compiled_material(
        transaction, instance.get(), module_filename, module_name,
        mdl_meters_per_scene_unit, mdl_wavelength_min, mdl_wavelength_max, load_resources);
}

namespace{

    void add_and_log_message(Execution_context* context, const Message& message, mi::Sint32 result)
    {
        context->add_message(message);
        context->add_error_message(message);
        context->set_result(result);
        LOG::mod_log->error(M_SCENE, LOG::Mod_log::C_DATABASE, "%s", message.m_message.c_str());
    }
};

const mi::mdl::IGenerated_code_dag::IMaterial_instance*
Mdl_material_instance::create_dag_material_instance(
    DB::Transaction* transaction,
    bool use_temporaries,
    bool class_compilation,
    Execution_context* context) const
{
    // get code DAG
    DB::Access<Mdl_module> module( m_module_tag, transaction);
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());

    // create new MDL material instance
    mi::mdl::IGenerated_code_dag::Error_code error_code;
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> instance(
        code_dag->create_material_instance( m_material_index, &error_code));
    ASSERT( M_SCENE, error_code == 0);
    ASSERT( M_SCENE, instance.is_valid_interface());

    mi::Float32 mdl_meters_per_scene_unit = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_METERS_PER_SCENE_UNIT);
    mi::Float32 mdl_wavelength_min = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_WAVELENGTH_MIN);
    mi::Float32 mdl_wavelength_max = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_WAVELENGTH_MAX);

    // convert m_arguments to DAG nodes
    mi::Uint32 n = code_dag->get_material_parameter_count( m_material_index);
    std::vector<const mi::mdl::DAG_node*> mdl_arguments( n);
    for( mi::Uint32 i = 0; i < n; ++i) {

        const mi::mdl::IType* parameter_type
            = code_dag->get_material_parameter_type( m_material_index, i);
        mi::base::Handle<const IExpression> argument( m_arguments->get_expression( i));
        mdl_arguments[i] = int_expr_to_mdl_dag_node(
            transaction, instance.get(), parameter_type, argument.get(), mdl_meters_per_scene_unit,
            mdl_wavelength_min, mdl_wavelength_max);
        if( !mdl_arguments[i]) {

            add_and_log_message(context, Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "Type mismatch, call of an unsuitable DB element, or call cycle in a graph rooted "
                "at the material definition \"" +
                add_mdl_db_prefix(code_dag->get_material_name(m_material_index)) + "\"."), -1);
            return 0;
        }
    }

    // initialize MDL material instance
    Call_evaluator    call_evaluator(
        transaction,
        context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES));
    Mdl_call_resolver resolver( transaction);

    mi::Uint32 flags = class_compilation
        ?
              mi::mdl::IGenerated_code_dag::IMaterial_instance::CLASS_COMPILATION
            | mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_RESOURCE_SHARING
            | mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_ARGUMENT_INLINE
        :
              mi::mdl::IGenerated_code_dag::IMaterial_instance::INSTANCE_COMPILATION;

    error_code = instance->initialize(
        &resolver,
        /*resource_modifier=*/NULL,
        code_dag.get(),
        n,
        n > 0 ? &mdl_arguments[0] : 0,
        use_temporaries,
        flags,
        class_compilation ? 0 : &call_evaluator,
        mdl_meters_per_scene_unit,
        mdl_wavelength_min, mdl_wavelength_max);
    switch( error_code) {
        case mi::mdl::IGenerated_code_dag::EC_NONE:
            break;
        case mi::mdl::IGenerated_code_dag::EC_ARGUMENT_TYPE_MISMATCH: {
            
            add_and_log_message(context, Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "Type mismatch for an argument in a graph rooted at the material "
                "definition \"" + 
                add_mdl_db_prefix( code_dag->get_material_name( m_material_index)) + "\".", 
                mi::mdl::IGenerated_code_dag::EC_ARGUMENT_TYPE_MISMATCH, 
                Message::MSG_COMPILER_DAG), -1);
            return 0;
        }
        case mi::mdl::IGenerated_code_dag::EC_WRONG_TRANSMISSION_ON_THIN_WALLED: {

            add_and_log_message(context, Message(mi::base::MESSAGE_SEVERITY_ERROR,
                "The thin-walled material instance rooted of the material definition \"" +
                add_mdl_db_prefix(code_dag->get_material_name(m_material_index)) + "\" has "
                "different transmission for surface and backface.",
                mi::mdl::IGenerated_code_dag::EC_WRONG_TRANSMISSION_ON_THIN_WALLED, 
                Message::MSG_COMPILER_DAG), -2);
            return 0;
        }
        case mi::mdl::IGenerated_code_dag::EC_INSTANTIATION_ERROR:
        case mi::mdl::IGenerated_code_dag::EC_INVALID_INDEX:
        case mi::mdl::IGenerated_code_dag::EC_MATERIAL_HAS_ERROR:
        case mi::mdl::IGenerated_code_dag::EC_TOO_FEW_ARGUMENTS:
        case mi::mdl::IGenerated_code_dag::EC_TOO_MANY_ARGUMENTS:
            ASSERT( M_SCENE, false);
            break;
    }

    const mi::mdl::Messages& msgs = instance->access_messages();
    report_messages( msgs, context);

    if (msgs.get_error_message_count() > 0) {
        context->set_result(-3);
        return 0;
    }
    instance->retain();
    return instance.get();
}

const mi::mdl::IType* Mdl_material_instance::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    DB::Access<Mdl_material_definition> definition( m_definition_tag, transaction);
    return definition.is_valid() ? definition->get_mdl_parameter_type( transaction, index) : 0;
}

void Mdl_material_instance::swap( Mdl_material_instance& other)
{
    SCENE::Scene_element<Mdl_material_instance, Mdl_material_instance::id>::swap( other);

    std::swap( m_module_tag, other.m_module_tag);
    std::swap( m_definition_tag, other.m_definition_tag);
    std::swap( m_material_index, other.m_material_index);
    m_definition_name.swap( other.m_definition_name);
    std::swap( m_immutable, other.m_immutable);

    std::swap( m_parameter_types, other.m_parameter_types);
    std::swap( m_arguments, other.m_arguments);
    std::swap( m_enable_if_conditions, other.m_enable_if_conditions);
}

const SERIAL::Serializable* Mdl_material_instance::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serializer->write( m_module_tag);
    serializer->write( m_definition_tag);
    serializer->write( m_material_index);
    serializer->write( m_definition_name);
    serializer->write( m_immutable);

    m_tf->serialize_list( serializer, m_parameter_types.get());
    m_ef->serialize_list( serializer, m_arguments.get());
    m_ef->serialize_list( serializer, m_enable_if_conditions.get());

    return this + 1;
}

SERIAL::Serializable* Mdl_material_instance::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_module_tag);
    deserializer->read( &m_definition_tag);
    deserializer->read( &m_material_index);
    deserializer->read( &m_definition_name);
    deserializer->read( &m_immutable);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    m_arguments = m_ef->deserialize_list( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    return this + 1;
}

void Mdl_material_instance::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    s << "MDL module tag: " << m_module_tag.get_uint() << std::endl;
    s << "Material definition tag: " << m_definition_tag.get_uint() << std::endl;
    s << "Material definition MDL name: \"" << m_definition_name << "\"" << std::endl;
    tmp = m_ef->dump( transaction, m_arguments.get(), /*name*/ 0);
    s << "Arguments: " << tmp->get_c_str() << std::endl;
    s << "Immutable: " << m_immutable << std::endl;
    tmp = m_ef->dump( transaction, m_enable_if_conditions.get(), /*name*/ 0);
    s << "Enable_if conditions: " << tmp->get_c_str() << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_material_instance::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_material_instance, Mdl_material_instance::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_material_instance, Mdl_material_instance::id>)
        + dynamic_memory_consumption( m_definition_name)
        + dynamic_memory_consumption( m_parameter_types)
        + dynamic_memory_consumption( m_arguments)
        + dynamic_memory_consumption( m_enable_if_conditions);
}

DB::Journal_type Mdl_material_instance::get_journal_flags() const
{
    return SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE;
}

Uint Mdl_material_instance::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_material_instance::get_scene_element_references( DB::Tag_set* result) const
{
    // default functions are held by the module, avoid cycle.
    if (!m_immutable) {
        ASSERT(M_SCENE, m_module_tag);
        result->insert(m_module_tag);
    }
    collect_references( m_arguments.get(), result);
    collect_references( m_enable_if_conditions.get(), result);
}

} // namespace MDL

} // namespace MI
