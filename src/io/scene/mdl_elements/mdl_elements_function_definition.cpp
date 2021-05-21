/***************************************************************************************************
 * Copyright (c) 2012-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_function_definition.h"

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_utilities.h"

#include <sstream>
#include <map>

#include <mi/neuraylib/istring.h>
#include <mi/mdl/mdl_archiver.h>
#include <base/system/main/access_module.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/hal/disk/disk.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace MDL {

Mdl_function_definition::Mdl_function_definition()
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_filename()
, m_module_mdl_name()
, m_module_db_name()
, m_function_tag()
, m_function_ident( -1)
, m_mdl_semantic( mi::mdl::IDefinition::DS_UNKNOWN)
, m_semantic( mi::neuraylib::IFunction_definition::DS_UNKNOWN)
, m_mdl_name()
, m_simple_name()
, m_db_name()
, m_original_name()
, m_prototype_tag()
, m_is_exported( false)
, m_is_uniform( false)
, m_is_material( false)
, m_since_version( mi_mdl_IMDL_MDL_VERSION_INVALID)
, m_removed_version( mi_mdl_IMDL_MDL_VERSION_INVALID)
, m_parameter_types()
, m_parameter_type_names()
, m_return_type()
, m_defaults()
, m_annotations()
, m_parameter_annotations()
, m_return_annotations()
, m_enable_if_conditions()
, m_enable_if_users()
, m_function_hash()
{
}

Mdl_function_definition::Mdl_function_definition(
    DB::Transaction* transaction,
    DB::Tag function_tag,
    Mdl_ident function_ident,
    const mi::mdl::IModule* module,
    const mi::mdl::IGenerated_code_dag* mdl_code_dag,
    bool is_material,
    mi::Size index,
    const char* module_filename,
    const char* module_mdl_name,
    bool load_resources)
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_filename( module_filename ? module_filename : "")
, m_module_mdl_name()
, m_module_db_name()
, m_function_tag( function_tag)
, m_function_ident( function_ident)
, m_original_name()
, m_prototype_tag()
, m_is_material( is_material)
, m_parameter_types()
, m_return_type()
, m_defaults()
, m_annotations()
, m_parameter_annotations()
, m_return_annotations()
, m_enable_if_conditions()
, m_enable_if_users()
, m_function_hash()
{
    m_module_mdl_name = module_mdl_name;
    m_module_db_name  = get_db_name( m_module_mdl_name);

    Code_dag code_dag( mdl_code_dag, m_is_material);

    m_simple_name  = code_dag.get_simple_name( index);
    m_mdl_name     = MDL::get_mdl_name( mdl_code_dag, is_material, index);
    m_db_name      = get_db_name( m_mdl_name);
    m_mdl_semantic = code_dag.get_semantics( index);
    m_semantic     = mdl_semantics_to_ext_semantics( m_mdl_semantic);
    m_is_exported  = code_dag.get_exported( index);
    m_is_uniform   = code_dag.get_uniform( index);

    const char* s = code_dag.get_cloned_name( index);
    std::string prototype_name = s ? s : "";
    if( !prototype_name.empty()) {
        prototype_name
            = encode_name_add_missing_signature( transaction, mdl_code_dag, prototype_name);
        m_prototype_tag = transaction->name_to_tag( get_db_name( prototype_name).c_str());
    }
    ASSERT( M_SCENE, m_prototype_tag || prototype_name.empty());

    const char* original_name = code_dag.get_original_name( index);
    if( original_name)
        m_original_name
            = encode_name_add_missing_signature( transaction, mdl_code_dag, original_name);

    m_tf = get_type_factory();
    m_vf = get_value_factory();
    m_ef = get_expression_factory();

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        mdl_code_dag->get_resource_tagger(),
        mdl_code_dag,
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        module_filename,
        m_module_mdl_name.c_str(),
        m_prototype_tag,
        load_resources,
        /*user_modules_seen*/ nullptr);

    // return type
    if( m_is_material)
        m_return_type = get_type_factory()->get_predefined_struct( IType_struct::SID_MATERIAL);
    else {
        const mi::mdl::IType* return_type = code_dag.get_return_type( index);
        m_return_type = mdl_type_to_int_type( m_tf.get(), return_type);
    }

    // function annotations
    mi::Size annotation_count = code_dag.get_annotation_count( index);
    Mdl_annotation_block annotations( annotation_count);
    for( mi::Size i = 0; i < annotation_count; ++i)
        annotations[i] = code_dag.get_annotation( index, i);
    m_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
         annotations, m_mdl_name.c_str());

    // return type annotations
    mi::Size return_annotation_count
        = code_dag.get_return_annotation_count( index);
    Mdl_annotation_block return_annotations( return_annotation_count);
    for( mi::Size i = 0; i < return_annotation_count; ++i)
        return_annotations[i] = code_dag.get_return_annotation( index, i);
    m_return_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
       return_annotations, m_mdl_name.c_str());

    // parameters/arguments
    m_defaults = m_ef->create_expression_list();
    m_parameter_annotations = m_ef->create_annotation_list();
    m_parameter_types = m_tf->create_type_list();
    m_enable_if_conditions = m_ef->create_expression_list();

    mi::Size parameter_count = code_dag.get_parameter_count( index);
    m_enable_if_users.resize( parameter_count);

    for( mi::Size i = 0; i < parameter_count; ++i) {

        const char* parameter_name = code_dag.get_parameter_name( index, i);

        // update m_parameter_types
        const mi::mdl::IType* parameter_type
            = code_dag.get_parameter_type( index, i);
        mi::base::Handle<const IType> type( mdl_type_to_int_type( m_tf.get(), parameter_type));
        m_parameter_types->add_type( parameter_name, type.get());
        m_parameter_type_names.emplace_back(
            encode_name_without_signature( code_dag.get_parameter_type_name( index, i)));

        // update m_defaults
        const mi::mdl::DAG_node* default_
            = code_dag.get_parameter_default( index, i);
        if( default_) {
            mi::base::Handle<IExpression> default_int( converter.mdl_dag_node_to_int_expr(
               default_, type.get()));
            ASSERT( M_SCENE, default_int);
            m_defaults->add_expression( parameter_name, default_int.get());
        }

        // update enable_if conditions
        const mi::mdl::DAG_node* enable_if_cond
            = code_dag.get_parameter_enable_if_condition( index, i);
        if( enable_if_cond) {
            mi::base::Handle<IExpression> enable_if_cond_int( converter.mdl_dag_node_to_int_expr(
                enable_if_cond, type.get()));
            ASSERT( M_SCENE, enable_if_cond_int);
            m_enable_if_conditions->add_expression( parameter_name, enable_if_cond_int.get());
        }
        std::vector<mi::Size> &users = m_enable_if_users[i];
        mi::Size n_users = code_dag.get_parameter_enable_if_condition_users(
            index, i);
        for( mi::Size j = 0; j < n_users; ++j) {
            mi::Size param_idx = code_dag.get_parameter_enable_if_condition_user(
                index, i, j);
            users.push_back( param_idx);
        }

        // update m_parameter_annotations
        mi::Size parameter_annotation_count
            = code_dag.get_parameter_annotation_count( index, i);
        Mdl_annotation_block parameter_annotations( parameter_annotation_count);
        for( mi::Size j = 0; j < parameter_annotation_count; ++j)
            parameter_annotations[j]
                = code_dag.get_parameter_annotation( index, i, j);
        mi::base::Handle<IAnnotation_block> block(
            converter.mdl_dag_node_vector_to_int_annotation_block(
                parameter_annotations, m_mdl_name.c_str()));
        if( block)
            m_parameter_annotations->add_annotation_block( parameter_name, block.get());
    }

    // hash
    if( const mi::mdl::DAG_hash* hash = code_dag.get_hash( index))
        m_function_hash = convert_hash( *hash);

    // MDL versions
    compute_mdl_version( module);
}

DB::Tag Mdl_function_definition::get_module( DB::Transaction* transaction) const
{
    return transaction->name_to_tag( m_module_db_name.c_str());
}

const char* Mdl_function_definition::get_mdl_name() const
{
    return m_mdl_name.c_str();
}

const char* Mdl_function_definition::get_mdl_module_name() const
{
    return m_module_mdl_name.c_str();
}

const char* Mdl_function_definition::get_mdl_simple_name() const
{
    return m_simple_name.c_str();
}

const char* Mdl_function_definition::get_mdl_parameter_type_name( mi::Size index) const
{
    if( index >= m_parameter_type_names.size())
        return nullptr;
    return m_parameter_type_names[index].c_str();
}

DB::Tag Mdl_function_definition::get_prototype() const
{
    return m_prototype_tag;
}

void Mdl_function_definition::get_mdl_version(
    mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const
{
    since   = MDL::convert_mdl_version( m_since_version);
    removed = MDL::convert_mdl_version( m_removed_version);
}

mi::neuraylib::IFunction_definition::Semantics Mdl_function_definition::get_semantic() const
{
    return m_semantic;
}

const IType* Mdl_function_definition::get_return_type() const
{
    m_return_type->retain();
    return m_return_type.get();
}

mi::Size Mdl_function_definition::get_parameter_count() const
{
    return m_parameter_types->get_size();
}

const char* Mdl_function_definition::get_parameter_name( mi::Size index) const
{
    return m_parameter_types->get_name( index);
}

mi::Size Mdl_function_definition::get_parameter_index( const char* name) const
{
    return m_parameter_types->get_index( name);
}

const IType_list* Mdl_function_definition::get_parameter_types() const
{
    m_parameter_types->retain();
    return m_parameter_types.get();
}

const IExpression_list* Mdl_function_definition::get_defaults() const
{
    m_defaults->retain();
    return m_defaults.get();
}

const IExpression_list* Mdl_function_definition::get_enable_if_conditions() const
{
    m_enable_if_conditions->retain();
    return m_enable_if_conditions.get();
}

mi::Size Mdl_function_definition::get_enable_if_users( mi::Size index) const
{
    if( index >= m_enable_if_users.size())
        return 0;
    return m_enable_if_users[index].size();
}

mi::Size Mdl_function_definition::get_enable_if_user( mi::Size index, mi::Size u_index) const
{
    if( index >= m_enable_if_users.size() || u_index >= m_enable_if_users[index].size())
        return ~mi::Size( 0);

    return m_enable_if_users[index][u_index];
}

const IAnnotation_block* Mdl_function_definition::get_annotations() const
{
    if( !m_annotations)
        return nullptr;
    m_annotations->retain();
    return m_annotations.get();
}

const IAnnotation_block* Mdl_function_definition::get_return_annotations() const
{
    if( !m_return_annotations)
        return nullptr;
    m_return_annotations->retain();
    return m_return_annotations.get();
}

const IAnnotation_list* Mdl_function_definition::get_parameter_annotations() const
{
    m_parameter_annotations->retain();
    return m_parameter_annotations.get();
}

const IExpression* Mdl_function_definition::get_body( DB::Transaction* transaction) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return 0;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return 0;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    const mi::mdl::DAG_node* body = code_dag.get_body( definition_index);
    if( !body)
        return nullptr;

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        mdl_code_dag->get_resource_tagger(),
        mdl_code_dag.get(),
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_filename*/ nullptr,
        /*module_mdl_name*/ nullptr,
        m_prototype_tag,
        /*load_resources*/ false,
        /*user_modules_seen*/ nullptr);

    mi::base::Handle<const IExpression> body_int( converter.mdl_dag_node_to_int_expr( body, nullptr));
    return body_int->get_interface<const IExpression>();
}

mi::Size Mdl_function_definition::get_temporary_count( DB::Transaction* transaction) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return 0;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return 0;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    return code_dag.get_temporary_count( definition_index);
}

const IExpression* Mdl_function_definition::get_temporary(
    DB::Transaction* transaction, mi::Size index) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return nullptr;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return nullptr;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    if( index >= code_dag.get_temporary_count( definition_index))
        return nullptr;

    const mi::mdl::DAG_node* temporary = code_dag.get_temporary( definition_index, index);

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        mdl_code_dag->get_resource_tagger(),
        mdl_code_dag.get(),
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_filename*/ nullptr,
        /*module_mdl_name*/ nullptr,
        m_prototype_tag,
        /*load_resources*/ false,
        /*user_modules_seen*/ nullptr);

    return converter.mdl_dag_node_to_int_expr( temporary, nullptr);
}

const char* Mdl_function_definition::get_temporary_name(
    DB::Transaction* transaction, mi::Size index) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return nullptr;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return nullptr;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    if( index >= code_dag.get_temporary_count( definition_index))
        return nullptr;

    return code_dag.get_temporary_name( definition_index, index);
}

std::string Mdl_function_definition::get_thumbnail() const
{
    if( !m_is_exported || m_module_filename.empty())
        return std::string();

    // See MDL-559 for reasons why this string is not computed once in the constructor.
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IArchive_tool> archive_tool( mdl->create_archive_tool());
    return DETAIL::lookup_thumbnail(
        m_module_filename, m_module_mdl_name, m_simple_name, m_annotations.get(), archive_tool.get());
}

Mdl_function_call* Mdl_function_definition::create_function_call(
   DB::Transaction* transaction, const IExpression_list* arguments, mi::Sint32* errors) const
{
    Execution_context context;
    if( !is_valid( transaction, &context)) {
        if( errors)
            *errors = -9;
        return nullptr;
    }

    switch( m_semantic) {
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
            return create_array_constructor_call_internal(
                transaction, arguments, /*allow_ek_parameter*/ false, /*immutable*/ false, errors);
        case mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX:
            return create_array_index_operator_call_internal(
                transaction, arguments, /*immutable*/ false, errors);
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
            return create_array_length_operator_call_internal(
                transaction, arguments, /*immutable*/ false, errors);
        case mi::neuraylib::IFunction_definition::DS_TERNARY:
            return create_ternary_operator_call_internal(
                transaction, arguments, /*immutable*/ false, errors);
        case mi::neuraylib::IFunction_definition::DS_CAST:
            return create_cast_operator_call_internal(
                transaction, arguments, /*immutable*/ false, errors);
        default:
            break;
    }

    return create_function_call_internal(
        transaction, arguments, /*allow_ek_parameter*/ false, /*immutable*/ false, errors);
}

IExpression_direct_call* Mdl_function_definition::create_direct_call(
    DB::Transaction* transaction, const IExpression_list* arguments, mi::Sint32* errors) const
{
    Execution_context context;
    if( !is_valid( transaction, &context)) {
        if( errors)
            *errors = -9;
        return nullptr;
    }

    switch( m_semantic) {
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
            return create_array_constructor_direct_call_internal(
                transaction, arguments, errors);
        case mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX:
            return create_array_index_operator_direct_call_internal(
                transaction, arguments, errors);
        case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
            return create_array_length_operator_direct_call_internal(
                transaction, arguments, errors);
        case mi::neuraylib::IFunction_definition::DS_TERNARY:
            return create_ternary_operator_direct_call_internal(
                transaction, arguments, errors);
        case mi::neuraylib::IFunction_definition::DS_CAST:
            return create_cast_operator_direct_call_internal(
                transaction, arguments, errors);
        default:
            break;
    }

    return create_direct_call_internal(
        transaction, arguments, errors);
}

Mdl_function_call* Mdl_function_definition::create_function_call_internal(
   DB::Transaction* transaction,
   const IExpression_list* arguments,
   bool allow_ek_parameter,
   bool immutable,
   mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> complete_arguments( check_and_prepare_arguments(
        transaction,
        arguments,
        allow_ek_parameter,
        /*allow_ek_direct_call*/ false,
        /*create_direct_calls*/ false,
        /*copy_immutable_calls*/ !immutable,
        errors));
    if( !complete_arguments)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        m_is_material,
        complete_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        m_parameter_types.get(),
        m_return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_direct_call_internal(
   DB::Transaction* transaction,
   const IExpression_list* arguments,
   mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> complete_arguments( check_and_prepare_arguments(
        transaction,
        arguments,
        /*allow_ek_parameter*/ true,
        /*allow_ek_direct_call*/ true,
        /*create_direct_calls*/ true,
        /*copy_immutable_calls*/ false,
        errors));
    if( !complete_arguments)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        m_return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        complete_arguments.get());

    *errors = 0;
    return direct_call;
}

Mdl_function_call* Mdl_function_definition::create_array_constructor_call_internal(
   DB::Transaction* transaction,
   const IExpression_list* arguments,
   bool allow_ek_parameter,
   bool immutable,
   mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_constructor_operator(
            transaction,
            arguments,
            allow_ek_parameter,
            /*allow_ek_direct_call*/ false,
            /*create_direct_calls*/ false,
            /*copy_immutable_calls*/ !immutable,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        /*m_is_material*/ false,
        new_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_array_constructor_direct_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_constructor_operator(
            transaction,
            arguments,
            /*allow_ek_parameter*/ true,
            /*allow_ek_direct_call*/ true,
            /*create_direct_calls*/ true,
            /*copy_immutable_calls*/ false,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        new_arguments.get());

    *errors = 0;
    return direct_call;
}

Mdl_function_call* Mdl_function_definition::create_array_index_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_index_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ !immutable,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        /*m_is_material*/ false,
        new_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_array_index_operator_direct_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_index_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ false,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        new_arguments.get());

    *errors = 0;
    return direct_call;
}

Mdl_function_call* Mdl_function_definition::create_array_length_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_length_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ !immutable,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        /*m_is_material*/ false,
        new_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_array_length_operator_direct_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_array_length_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ false,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        new_arguments.get());

    *errors = 0;
    return direct_call;
}

Mdl_function_call* Mdl_function_definition::create_ternary_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_ternary_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ !immutable,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        /*m_is_material*/ false,
        new_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_ternary_operator_direct_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_ternary_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ false,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        new_arguments.get());

    *errors = 0;
    return direct_call;
}

Mdl_function_call* Mdl_function_definition::create_cast_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_cast_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ !immutable,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module( transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        /*m_is_material*/ false,
        new_arguments.get(),
        m_mdl_semantic,
        m_mdl_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

IExpression_direct_call* Mdl_function_definition::create_cast_operator_direct_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    mi::base::Handle<IExpression_list> new_arguments;
    mi::base::Handle<IType_list> parameter_types;
    mi::base::Handle<const IType> return_type;
    std::tie( new_arguments, parameter_types, return_type)
        = check_and_prepare_arguments_cast_operator(
            transaction,
            arguments,
            /*copy_immutable_calls*/ false,
            errors);
    if( !new_arguments || !parameter_types || !return_type)
        return nullptr;

    IExpression_direct_call* direct_call = m_ef->create_direct_call(
        return_type.get(),
        transaction->name_to_tag( m_module_db_name.c_str()),
        {m_function_tag, m_function_ident},
        m_db_name.c_str(),
        new_arguments.get());

    *errors = 0;
    return direct_call;
}

mi::mdl::IDefinition::Semantics Mdl_function_definition::get_mdl_semantic() const
{
    return m_mdl_semantic;
}

const mi::mdl::IType* Mdl_function_definition::get_mdl_return_type(
    DB::Transaction* transaction) const
{
    // DS_INTRINSIC_DAG_ARRAY_LENGTH is template-like, but the return type is fixed
    ASSERT( M_SCENE,
           m_semantic != mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        && m_semantic != mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX
        && m_semantic != mi::neuraylib::IFunction_definition::DS_CAST
        && m_semantic != mi::neuraylib::IFunction_definition::DS_TERNARY
        && "return type for template-like function must be calculated");

    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return 0;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return 0;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    return code_dag.get_return_type( definition_index);
}

const mi::mdl::IType* Mdl_function_definition::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    ASSERT( M_SCENE,
            m_semantic != mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        &&  m_semantic != mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH
        && (m_semantic != mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX || index == 1)
        &&  m_semantic != mi::neuraylib::IFunction_definition::DS_CAST
        && (m_semantic != mi::neuraylib::IFunction_definition::DS_TERNARY || index == 0)
        && "parameter type for template-like function must be calculated");

    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context*/ nullptr))
        return 0;
    if( module->has_definition( m_is_material,  m_db_name, m_function_ident) != 0)
        return 0;

    mi::Size definition_index
        = module->get_definition_index( m_is_material, m_db_name, m_function_ident);
    ASSERT( M_SCENE, definition_index != ~mi::Size( 0));

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> mdl_code_dag( module->get_code_dag());
    Code_dag code_dag( mdl_code_dag.get(), m_is_material);
    return code_dag.get_parameter_type( definition_index, index);
}

std::string Mdl_function_definition::get_mdl_name_without_parameter_types() const
{
    return m_module_mdl_name == get_builtins_module_mdl_name()
        ? std::string( "::") + m_simple_name
        : m_module_mdl_name + "::" + m_simple_name;
}

const char* Mdl_function_definition::get_mdl_original_name() const
{
    return m_original_name.empty() ? nullptr : m_original_name.c_str();
}

const char* Mdl_function_definition::get_module_db_name() const
{
    ASSERT(M_SCENE, !m_module_db_name.empty());
    return m_module_db_name.c_str();
}

bool Mdl_function_definition::is_valid(
    DB::Transaction* transaction,
    Execution_context* context) const
{
    DB::Tag module_tag = get_module(transaction);
    DB::Access<Mdl_module> module(module_tag, transaction);
    if (!module->is_valid(transaction, context))
        return false;

    if (module->has_definition( m_is_material, m_db_name, m_function_ident) < 0)
        return false;

    // check defaults. is this really needed?
    for (mi::Size i = 0; i < m_defaults->get_size(); ++i) {

        mi::base::Handle<const IExpression_call> expr(
            m_defaults->get_expression<IExpression_call>(i));
        if (!expr)
            continue;
        DB::Tag call_tag = expr->get_call();
        if (!call_tag)
            continue;
        SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
        if (class_id != ID_MDL_FUNCTION_CALL) {
            add_context_error(
                context, "The function call attached to parameter '"
                + std::string(m_defaults->get_name(i)) + "' has a wrong element type.", -1);
            return false;
        }
        DB::Access<Mdl_function_call> fcall(call_tag, transaction);
        DB::Tag_set tags_seen;
        if (!fcall->is_valid(transaction, tags_seen, context))
            return false;
    }
    return true;
}

namespace
{
    bool is_compatible_annotation_block(
        const IExpression_factory* ef,
        const IAnnotation_block* rhs,
        const IAnnotation_block* lhs)
    {
        if (lhs->get_size() != rhs->get_size())
            return false;

        // for now, annotations must match. this implies, that all annotations
        // used by this definition that are part of the module the definition
        // comes from, also still exist.
        for (mi::Size i = 0, n = rhs->get_size(); i < n; ++i) {

            mi::base::Handle<const IAnnotation> lhs_anno(lhs->get_annotation(i));
            mi::base::Handle<const IAnnotation> rhs_anno(rhs->get_annotation(i));

            const char* lhs_anno_name = lhs_anno->get_name();
            const char* rhs_anno_name = rhs_anno->get_name();
            if (strcmp(lhs_anno_name, rhs_anno_name) != 0)
                return false;

            mi::base::Handle<const IExpression_list> lhs_anno_args(lhs_anno->get_arguments());
            mi::base::Handle<const IExpression_list> rhs_anno_args(rhs_anno->get_arguments());

            if (ef->compare(lhs_anno_args.get(), rhs_anno_args.get()) != 0)
                return false;
        }

        return true;
    }
} // anonymous

bool Mdl_function_definition::is_compatible(const Mdl_function_definition& other) const
{
    if (m_semantic != other.m_semantic)
        return false;

    if (m_is_exported != other.m_is_exported)
        return false;

    if (m_is_uniform != other.m_is_uniform)
        return false;

    if (m_prototype_tag != other.m_prototype_tag)
        return false;

    if (m_original_name != other.m_original_name)
        return false;

    if (m_tf->compare(m_return_type.get(), other.m_return_type.get()) != 0)
        return false;

    if (m_tf->compare(m_parameter_types.get(), other.m_parameter_types.get()) != 0)
        return false;

    if (m_ef->compare(m_defaults.get(), other.m_defaults.get()) != 0)
        return false;

    // check annotations
    if (m_annotations.is_valid_interface() != other.m_annotations.is_valid_interface())
        return false;

    if (m_annotations) {
        if (!is_compatible_annotation_block(
            m_ef.get(), m_annotations.get(), other.m_annotations.get())) {
            return false;
        }
    }

    // check parameter annotations
    if (m_parameter_annotations.is_valid_interface() !=
        other.m_parameter_annotations.is_valid_interface()) {
        return false;
    }

    if (m_parameter_annotations) {
        if (m_parameter_annotations->get_size() != other.m_parameter_annotations->get_size())
            return false;

        for (mi::Size i = 0, n = m_parameter_annotations->get_size(); i < n; ++i) {
            const char* block_name = m_parameter_annotations->get_name(i);
            if (!block_name)
                return false;

            mi::base::Handle<const IAnnotation_block> param_block_anno(
                m_parameter_annotations->get_annotation_block(block_name));

            mi::base::Handle<const IAnnotation_block> other_param_block_anno(
                other.m_parameter_annotations->get_annotation_block(block_name));
            if (!other_param_block_anno)
                return false; // no block with matching name

            if (!is_compatible_annotation_block(
                m_ef.get(), param_block_anno.get(), other_param_block_anno.get())) {
                return false;
            }
        }
    }

    // check return annotations
    if (m_return_annotations.is_valid_interface() != other.m_return_annotations.is_valid_interface())
        return false;

    if (m_return_annotations) {
        if (!is_compatible_annotation_block(
            m_ef.get(), m_return_annotations.get(), other.m_return_annotations.get())) {
            return false;
        }
    }
    return true;
}

Mdl_ident Mdl_function_definition::get_ident() const
{
    return m_function_ident;
}

void Mdl_function_definition::compute_mdl_version( const mi::mdl::IModule* mdl_module)
{
    const mi::mdl::Module* impl = mi::mdl::impl_cast<mi::mdl::Module>( mdl_module);

    if( m_is_material || (!mdl_module->is_stdlib() && !mdl_module->is_builtins())) {
        m_since_version   = impl->get_mdl_version();
        m_removed_version = mi_mdl_IMDL_MDL_VERSION_INVALID;
        return;
    }

    if( m_semantic == mi::neuraylib::IFunction_definition::DS_CAST) {
        m_since_version   = mi::mdl::IMDL::MDL_VERSION_1_5;
        m_removed_version = mi_mdl_IMDL_MDL_VERSION_INVALID;
        return;
    }

    if( mi::mdl::semantic_is_operator( m_mdl_semantic)
        || (   m_semantic >= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_FIRST
            && m_semantic <= mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_LAST)) {
        m_since_version   = mi::mdl::IMDL::MDL_VERSION_1_0;
        m_removed_version = mi_mdl_IMDL_MDL_VERSION_INVALID;
        return;
    }

    const mi::mdl::Definition* def = mi::mdl::impl_cast<mi::mdl::Definition>( impl->find_signature(
        m_mdl_name.c_str(), /*only_exported*/ !mdl_module->is_builtins()));
    if( !def) {
        ASSERT( M_SCENE, !"definition not found");
        m_since_version   = impl->get_mdl_version();
        m_removed_version = mi_mdl_IMDL_MDL_VERSION_INVALID;
        return;
    }

    unsigned flags = def->get_version_flags();
    m_since_version   = static_cast<mi::mdl::IMDL::MDL_version>( mi::mdl::mdl_since_version( flags));
    m_removed_version = static_cast<mi::mdl::IMDL::MDL_version>( mi::mdl::mdl_removed_version( flags));
}

void Mdl_function_definition::get_mdl_version(
    mi::mdl::IMDL::MDL_version& since, mi::mdl::IMDL::MDL_version& removed) const
{
    since   = m_since_version;
    removed = m_removed_version;
}

const SERIAL::Serializable* Mdl_function_definition::serialize(
    SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, m_module_filename);
    SERIAL::write( serializer, m_module_mdl_name);
    SERIAL::write( serializer, m_module_db_name);
    SERIAL::write( serializer, m_function_tag);
    SERIAL::write( serializer, m_function_ident);
    SERIAL::write_enum( serializer, m_mdl_semantic);
    SERIAL::write_enum( serializer, m_semantic);
    SERIAL::write( serializer, m_mdl_name);
    SERIAL::write( serializer, m_simple_name);
    SERIAL::write( serializer, m_db_name);
    SERIAL::write( serializer, m_original_name);
    SERIAL::write( serializer, m_prototype_tag);
    SERIAL::write( serializer, m_is_exported);
    SERIAL::write( serializer, m_is_uniform);
    SERIAL::write( serializer, m_is_material);
    SERIAL::write_enum( serializer, m_since_version);
    SERIAL::write_enum( serializer, m_removed_version);

    m_tf->serialize_list( serializer, m_parameter_types.get());
    SERIAL::write( serializer, m_parameter_type_names);
    m_tf->serialize( serializer, m_return_type.get());
    m_ef->serialize_list( serializer, m_defaults.get());
    m_ef->serialize_annotation_block( serializer, m_annotations.get());
    m_ef->serialize_annotation_list( serializer, m_parameter_annotations.get());
    m_ef->serialize_annotation_block( serializer, m_return_annotations.get());
    m_ef->serialize_list(serializer, m_enable_if_conditions.get());

    SERIAL::write( serializer, m_enable_if_users);
    write( serializer, m_function_hash);

    return this + 1;
}

SERIAL::Serializable* Mdl_function_definition::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    SERIAL::read( deserializer, &m_module_filename);
    if( !m_module_filename.empty()) {
        m_module_filename = HAL::Ospath::convert_to_platform_specific_path( m_module_filename);
        if( !DISK::access( m_module_filename.c_str()))
            m_module_filename.clear();
    }

    SERIAL::read( deserializer, &m_module_mdl_name);
    SERIAL::read( deserializer, &m_module_db_name);
    SERIAL::read( deserializer, &m_function_tag);
    SERIAL::read( deserializer, &m_function_ident);
    SERIAL::read_enum( deserializer, &m_mdl_semantic);
    SERIAL::read_enum( deserializer, &m_semantic);
    SERIAL::read( deserializer, &m_mdl_name);
    SERIAL::read( deserializer, &m_simple_name);
    SERIAL::read( deserializer, &m_db_name);
    SERIAL::read( deserializer, &m_original_name);
    SERIAL::read( deserializer, &m_prototype_tag);
    SERIAL::read( deserializer, &m_is_exported);
    SERIAL::read( deserializer, &m_is_uniform);
    SERIAL::read( deserializer, &m_is_material);
    SERIAL::read_enum( deserializer, &m_since_version);
    SERIAL::read_enum( deserializer, &m_removed_version);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    SERIAL::read( deserializer, &m_parameter_type_names);
    m_return_type = m_tf->deserialize( deserializer);
    m_defaults = m_ef->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_parameter_annotations = m_ef->deserialize_annotation_list( deserializer);
    m_return_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    SERIAL::read( deserializer, &m_enable_if_users);
    read( deserializer, &m_function_hash);

    return this + 1;
}

void Mdl_function_definition::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    s << "Module filename: " << m_module_filename << std::endl;
    s << "Module MDL name: " << m_module_mdl_name << std::endl;
    s << "Module DB name: " << m_module_db_name << std::endl;
    s << "Function definition tag: " << m_function_tag.get_uint() << std::endl;
    s << "Function definition ID: " << m_function_ident << std::endl;
    s << "Function MDL semantic: " << m_mdl_semantic << std::endl;
    s << "Function semantic: " << m_semantic << std::endl;
    s << "Function definition MDL name: " << m_mdl_name << std::endl;
    s << "Function definition MDL simple name: " << m_simple_name << std::endl;
    s << "Function definition DB name: "  << m_db_name << std::endl;
    s << "Function definition MDL original name: " << m_original_name << std::endl;
    s << "Prototype tag: " << m_prototype_tag.get_uint() << std::endl;
    s << "Is exported: " << m_is_exported << std::endl;
    s << "Is uniform: " << m_is_uniform << std::endl;
    s << "Is material: " << m_is_material << std::endl;
    s << "Since version: " << m_since_version << std::endl;
    s << "Removed version: " << m_removed_version << std::endl;

    tmp = m_tf->dump( m_parameter_types.get());
    s << "Parameter types: " << tmp->get_c_str() << std::endl;
    tmp = m_tf->dump( m_return_type.get());
    s << "Return type: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_defaults.get(), /*name*/ nullptr);
    s << "Defaults: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_annotations.get(), /*name*/ nullptr);
    s << "Annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_parameter_annotations.get(), /*name*/ nullptr);
    s << "Parameter annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_return_annotations.get(), /*name*/ nullptr);
    s << "Return annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_enable_if_conditions.get(), /*name*/ nullptr);
    s << "Enable_if conditions: " << tmp->get_c_str() << std::endl;
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_function_definition::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_function_definition, Mdl_function_definition::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_function_definition, Mdl_function_definition::id>)
        + dynamic_memory_consumption( m_module_filename)
        + dynamic_memory_consumption( m_module_mdl_name)
        + dynamic_memory_consumption( m_module_db_name)
        + dynamic_memory_consumption( m_mdl_name)
        + dynamic_memory_consumption( m_simple_name)
        + dynamic_memory_consumption( m_db_name)
        + dynamic_memory_consumption( m_original_name)
        + dynamic_memory_consumption( m_parameter_types)
        + dynamic_memory_consumption( m_parameter_type_names)
        + dynamic_memory_consumption( m_return_type)
        + dynamic_memory_consumption( m_defaults)
        + dynamic_memory_consumption( m_annotations)
        + dynamic_memory_consumption( m_parameter_annotations)
        + dynamic_memory_consumption( m_return_annotations)
        + dynamic_memory_consumption( m_enable_if_conditions);
}

DB::Journal_type Mdl_function_definition::get_journal_flags() const
{
    return DB::JOURNAL_NONE;
}

Uint Mdl_function_definition::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_function_definition::get_scene_element_references( DB::Tag_set* result) const
{
    // skip m_function_tag (own tag)
    collect_references( m_defaults.get(), result);
    collect_references( m_annotations.get(), result);
    collect_references( m_parameter_annotations.get(), result);
    collect_references( m_return_annotations.get(), result);
    collect_references( m_enable_if_conditions.get(), result);
}

IExpression_list* Mdl_function_definition::check_and_prepare_arguments(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool allow_ek_parameter,
    bool allow_ek_direct_call,
    bool create_direct_calls,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the general case
    ASSERT( M_SCENE,
           m_semantic != mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        && m_semantic != mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH
        && m_semantic != mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX
        && m_semantic != mi::neuraylib::IFunction_definition::DS_CAST
        && m_semantic != mi::neuraylib::IFunction_definition::DS_TERNARY);

    // prevent instantiation of non-exported function definitions
    if( !m_is_exported) {
        *errors = -4;
        return nullptr;
    }

    std::map<std::string, bool> needs_cast;
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // check that the provided arguments are parameters of the function definition and that their
    // types match the expected types
    if( arguments) {
        mi::Size n = arguments->get_size();
        for( mi::Size i = 0; i < n; ++i) {
            const char* name = arguments->get_name( i);
            mi::Size parameter_index = get_parameter_index( name);
            mi::base::Handle<const IType> expected_type(
                m_parameter_types->get_type( parameter_index));
            if( !expected_type) {
                *errors = -1;
                return nullptr;
            }
            mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
            mi::base::Handle<const IType> actual_type( argument->get_type());

            bool needs_cast_tmp = false;
            if( !argument_type_matches_parameter_type(
                m_tf.get(),
                actual_type.get(),
                expected_type.get(),
                allow_cast,
                needs_cast_tmp)) {
                *errors = -2;
                return nullptr;
            }
            needs_cast[name] = needs_cast_tmp;

            bool actual_type_varying
                = (actual_type->get_all_type_modifiers()   & IType::MK_VARYING) != 0;
            bool expected_type_uniform
                = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
            if( actual_type_varying && expected_type_uniform) {
                *errors = -5;
                return nullptr;
            }

            IExpression::Kind kind = argument->get_kind();
            if(     kind != IExpression::EK_CONSTANT
                &&  kind != IExpression::EK_CALL
                && (kind != IExpression::EK_PARAMETER   || !allow_ek_parameter)
                && (kind != IExpression::EK_DIRECT_CALL || !allow_ek_direct_call)) {
                *errors = -6;
                return nullptr;
            }

            if( expected_type_uniform && return_type_is_varying( transaction, argument.get())) {
                *errors = -8;
                return nullptr;
            }
        }
    }

    // build up complete argument set using the defaults where necessary
    mi::base::Handle<IExpression_list> complete_arguments( m_ef->create_expression_list());
    std::vector<mi::base::Handle<const IExpression> > call_context;
    for( mi::Size i = 0, n = m_parameter_types->get_size(); i < n;  ++i) {
        const char* name = get_parameter_name( i);
        mi::base::Handle<const IExpression> argument(
            arguments ? arguments->get_expression( name) : nullptr);
        if( argument) {
            // use provided argument
            mi::base::Handle<IExpression> argument_copy( m_ef->clone(
                argument.get(), transaction, copy_immutable_calls));
            ASSERT( M_SCENE, argument_copy);

            if( needs_cast[name]) {
                mi::base::Handle<const IType> expected_type(
                    m_parameter_types->get_type( i));
                mi::Sint32 errors = 0;
                argument_copy = m_ef->create_cast(
                    transaction,
                    argument_copy.get(),
                    expected_type.get(),
                    /*cast_db_name*/ nullptr,
                    /*force_cast*/ false,
                    create_direct_calls,
                    &errors);
                ASSERT( M_SCENE, argument_copy); // should always succeed.
            }
            argument = argument_copy;

        } else {
            // no argument provided, use default
            mi::base::Handle<const IExpression> default_( m_defaults->get_expression( name));
            if( !default_) {
                // no argument provided, no default available
                *errors = -3;
                return nullptr;
            }
            if( create_direct_calls) {
                Execution_context context;
                default_ = int_expr_call_to_int_expr_direct_call(
                    transaction, m_ef.get(), default_.get(), /*parameters*/ {}, &context);
            }
            mi::base::Handle<const IType> expected_type( m_parameter_types->get_type( name));
            bool expected_type_uniform
                = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
            if( expected_type_uniform && return_type_is_varying( transaction, default_.get())) {
                *errors = -8;
                return nullptr;
            }
            // use clone of default as argument
            mi::base::Handle<IExpression> default_copy(
                deep_copy( m_ef.get(), transaction, default_.get(), call_context));
            ASSERT( M_SCENE, default_copy);
            argument = default_copy;
        }
        complete_arguments->add_expression( name, argument.get());
        call_context.push_back( argument);
    }

   complete_arguments->retain();
   return complete_arguments.get();
}

std::tuple<IExpression_list*,IType_list*,const IType*>
Mdl_function_definition::check_and_prepare_arguments_cast_operator(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the cast operator
    ASSERT( M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_CAST);

    // the  cast operator is always exported
    ASSERT( M_SCENE, m_is_exported);

    // arguments are required
    if( !arguments) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // we need exactly two arguments
    mi::Size n = arguments->get_size();
    if( n != 2) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // check names
    mi::Size index_cast        = arguments->get_index( "cast");
    mi::Size index_cast_return = arguments->get_index( "cast_return");
    if(    index_cast        == static_cast<mi::Size>( -1)
        || index_cast_return == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IExpression> cast_from(
        arguments->get_expression( index_cast));
    mi::base::Handle<const IType> cast_from_type( cast_from->get_type());

    mi::base::Handle<const IExpression> cast_to(
        arguments->get_expression( index_cast_return));
    mi::base::Handle<const IType> cast_to_type( cast_to->get_type());

    if( m_tf->is_compatible( cast_from_type.get(), cast_to_type.get()) < 0) {
        *errors = -2;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // the actual call only has one argument, clone it and create a new list
    mi::base::Handle<IExpression_list> new_arguments( m_ef->create_expression_list());
    mi::base::Handle<IExpression> cast_from_copy(
        m_ef->clone( cast_from.get(), /*transaction*/ transaction, copy_immutable_calls));
    new_arguments->add_expression( "cast", cast_from_copy.get());

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    parameter_types->add_type( "cast", cast_from_type.get());

    new_arguments->retain();
    parameter_types->retain();
    cast_to_type->retain();
    return std::make_tuple( new_arguments.get(), parameter_types.get(), cast_to_type.get());
}

std::tuple<IExpression_list*,IType_list*,const IType*>
Mdl_function_definition::check_and_prepare_arguments_ternary_operator(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the ternary operator
    ASSERT( M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_TERNARY);

    // the ternary operator is always exported
    ASSERT( M_SCENE, m_is_exported);

    // arguments are required
    if( !arguments) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // we need exactly three arguments
    mi::Size n = arguments->get_size();
    if( n != 3) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // check names
    mi::Size index_cond      = arguments->get_index( "cond");
    mi::Size index_true_exp  = arguments->get_index( "true_exp");
    mi::Size index_false_exp = arguments->get_index( "false_exp");
    if(    index_cond      == static_cast<mi::Size>( -1)
        || index_true_exp  == static_cast<mi::Size>( -1)
        || index_false_exp == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IExpression> cond(
        arguments->get_expression( index_cond));
    mi::base::Handle<const IType> cond_type( cond->get_type());

    mi::base::Handle<const IExpression> true_exp(
        arguments->get_expression( index_true_exp));
    mi::base::Handle<const IType> true_type( true_exp->get_type());

    mi::base::Handle<const IExpression> false_exp(
        arguments->get_expression( index_false_exp));
    mi::base::Handle<const IType> false_type( false_exp->get_type());

    if( m_tf->compare( true_type.get(), false_type.get()) != 0) {
        *errors = -2;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // clone arguments
    mi::base::Handle<IExpression_list> new_arguments(
        m_ef->clone( arguments, transaction, copy_immutable_calls));

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    parameter_types->add_type( "cond", cond_type.get());
    parameter_types->add_type( "true_exp", true_type.get());
    parameter_types->add_type( "false_exp", false_type.get());

    new_arguments->retain();
    parameter_types->retain();
    true_type->retain();
    return std::make_tuple( new_arguments.get(), parameter_types.get(), true_type.get());
}

std::tuple<IExpression_list*,IType_list*,const IType*>
Mdl_function_definition::check_and_prepare_arguments_array_index_operator(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the array index operator
    ASSERT( M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX);

    // the index operator is always exported
    ASSERT( M_SCENE, m_is_exported);

    // arguments are required
    if( !arguments) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // we need exactly two arguments
    mi::Size n = arguments->get_size();
    if( n != 2) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // check names
    mi::Size index_a = arguments->get_index( "a");
    mi::Size index_i = arguments->get_index( "i");
    if(    index_a == static_cast<mi::Size>( -1)
        || index_i == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IExpression> base_expr(
        arguments->get_expression( index_a));
    mi::base::Handle<const IType> base_type( base_expr->get_type());
    mi::base::Handle<const IType> s_base_type( base_type->skip_all_type_aliases());

    IType::Kind base_kind = s_base_type->get_kind();
    if(    base_kind != IType::TK_ARRAY
        && base_kind != IType::TK_VECTOR
        && base_kind != IType::TK_MATRIX) {
        *errors = -2;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IExpression> index_expr(
        arguments->get_expression( index_i));
    mi::base::Handle<const IType> index_type( index_expr->get_type());
    mi::base::Handle<const IType> s_index_type( index_type->skip_all_type_aliases());

    if( s_index_type->get_kind() != IType::TK_INT) {
        *errors = -2;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IType_compound> s_base_type_compound(
        s_base_type->get_interface<IType_compound>());
    mi::base::Handle<const IType> ret_type( s_base_type_compound->get_component_type( 0));

    // result type is uniform if array and index type are uniform
    if(    (base_type->get_all_type_modifiers()  & IType::MK_UNIFORM) != 0
        && (index_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0) {
        IType_factory *tf = get_type_factory();
        ret_type = mi::base::make_handle( tf->create_alias( ret_type.get(), IType::MK_UNIFORM, nullptr));
    }

    // clone arguments
    mi::base::Handle<IExpression_list> new_arguments(
        m_ef->clone( arguments, transaction, copy_immutable_calls));

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    parameter_types->add_type( "a", base_type.get());
    parameter_types->add_type( "i", index_type.get());

    new_arguments->retain();
    parameter_types->retain();
    ret_type->retain();
    return std::make_tuple( new_arguments.get(), parameter_types.get(), ret_type.get());
}

std::tuple<IExpression_list*,IType_list*,const IType*>
Mdl_function_definition::check_and_prepare_arguments_array_length_operator(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the array length operator
    ASSERT(
        M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH);

    // the array length operator is always exported
    ASSERT( M_SCENE, m_is_exported);

    // arguments are required
    if( !arguments) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // we need exactly one argument
    mi::Size n = arguments->get_size();
    if( n != 1) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // check names
    mi::Size index_a = arguments->get_index( "a");
    if( index_a == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    mi::base::Handle<const IExpression> array_expr(
        arguments->get_expression( index_a));
    mi::base::Handle<const IType> array_type( array_expr->get_type());
    mi::base::Handle<const IType> s_array_type( array_type->skip_all_type_aliases());

    if( s_array_type->get_kind() != IType::TK_ARRAY) {
        *errors = -2;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // result type is "uniform int"
    IType_factory *tf = get_type_factory();
    mi::base::Handle<const IType> ret_type( tf->create_int());
    ret_type = mi::base::make_handle( tf->create_alias( ret_type.get(), IType::MK_UNIFORM, nullptr));

    // clone arguments
    mi::base::Handle<IExpression_list> new_arguments(
        m_ef->clone( arguments, transaction, copy_immutable_calls));

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    parameter_types->add_type( "a", array_type.get());

    new_arguments->retain();
    parameter_types->retain();
    ret_type->retain();
    return std::make_tuple( new_arguments.get(), parameter_types.get(), ret_type.get());
}

std::tuple<IExpression_list*,IType_list*,const IType*>
Mdl_function_definition::check_and_prepare_arguments_array_constructor_operator(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool allow_ek_parameter,
    bool allow_ek_direct_call,
    bool create_direct_calls,
    bool copy_immutable_calls,
    mi::Sint32* errors) const
{
    // check that this method is only used for the array constructor
    ASSERT( M_SCENE, m_mdl_semantic == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR);

    // the array constructor is always exported
    ASSERT( M_SCENE, m_is_exported);

    // arguments are required
    if( !arguments) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    // the array constructor needs at least one argument
    mi::Size n = arguments->get_size();
    if( n == 0) {
        *errors = -3;
        return std::make_tuple( nullptr, nullptr, nullptr);
    }

    std::map<std::string, bool> needs_cast;
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // check that the provided arguments are all of the same type
    mi::base::Handle<const IExpression> first_argument(
        arguments->get_expression( static_cast<mi::Size>( 0)));
    mi::base::Handle<const IType> expected_type( first_argument->get_type());
    bool expected_type_uniform
        = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;

    for( mi::Size i = 0; i < n; ++i) {

        // Unique names are guaranteed by the expression list. If this check is passed for
        // elements, then we have the names "0", "1", ... "n-1" (not necessarily in this order).
        const char* name = arguments->get_name( i);
        STLEXT::Likely<mi::Size> name_likely = STRING::lexicographic_cast_s<mi::Size>( name);
        if( !name_likely.get_status() || *name_likely.get_ptr() >= n) {
            *errors = -3;
            return std::make_tuple( nullptr, nullptr, nullptr);
        }

        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        mi::base::Handle<const IType> actual_type( argument->get_type());

        bool needs_cast_tmp = false;
        mi::Sint32 r = m_tf->is_compatible( actual_type.get(), expected_type.get());
        if( allow_cast) {
            if( r == 0) // compatible types
                needs_cast_tmp = true;
            else if( r < 0) {
                *errors = -2;
                return std::make_tuple( nullptr, nullptr, nullptr);
            }
        } else {
            if( r != 1) { // different types
                *errors = -2;
                return std::make_tuple( nullptr, nullptr, nullptr);
            }
        }
        needs_cast[name] = needs_cast_tmp;

        bool actual_type_varying
            = (actual_type->get_all_type_modifiers() & IType::MK_VARYING) != 0;
        if( actual_type_varying && expected_type_uniform) {
            *errors = -5;
            return std::make_tuple( nullptr, nullptr, nullptr);
        }

        IExpression::Kind kind = argument->get_kind();
        if(     kind != IExpression::EK_CONSTANT
            &&  kind != IExpression::EK_CALL
            && (kind != IExpression::EK_PARAMETER   || !allow_ek_parameter)
            && (kind != IExpression::EK_DIRECT_CALL || !allow_ek_direct_call)) {
            *errors = -6;
            return std::make_tuple( nullptr, nullptr, nullptr);
        }

        if( expected_type_uniform && return_type_is_varying( transaction, argument.get())) {
            *errors = -8;
            return std::make_tuple( nullptr, nullptr, nullptr);
        }
    }

    // clone arguments, create parameter types
    mi::base::Handle<IExpression_list> new_arguments( m_ef->create_expression_list());
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    for( mi::Size i = 0; i < n; ++i) {
        std::string name = std::to_string( i);
        mi::base::Handle<const IExpression> arg( arguments->get_expression( name.c_str()));
        mi::base::Handle<IExpression> new_arg(
            m_ef->clone( arg.get(), transaction, copy_immutable_calls));
        if( needs_cast[name])
            new_arg = m_ef->create_cast(
                transaction,
                new_arg.get(),
                expected_type.get(),
                /*cast_db_name*/ nullptr,
                /*force_cast*/ false,
                create_direct_calls,
                errors);
        new_arguments->add_expression( name.c_str(), new_arg.get());
        parameter_types->add_type( name.c_str(), expected_type.get());
    }

    // compute return type
    mi::base::Handle<const IType> return_type(
        m_tf->create_immediate_sized_array( expected_type.get(), n));

    new_arguments->retain();
    parameter_types->retain();
    return_type->retain();
    return std::make_tuple( new_arguments.get(), parameter_types.get(), return_type.get());
}

} // namespace MDL

} // namespace MI
