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

#include "pch.h"

#include "i_mdl_elements_function_definition.h"

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_material_instance.h"
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
, m_module_mdl_name()
, m_module_db_name()
, m_function_tag()
, m_function_ident( -1)
, m_mdl_semantic( mi::mdl::IDefinition::DS_UNKNOWN)
, m_semantic( mi::neuraylib::IFunction_definition::DS_UNKNOWN)
, m_name()
, m_simple_name()
, m_db_name()
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported( false)
, m_is_uniform( false)
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
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Size function_index,
    const char* module_filename,
    const char* module_name,
    bool load_resources)
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_mdl_name( module_name)
, m_module_db_name( get_db_name( module_name))
, m_function_tag( function_tag)
, m_function_ident( function_ident)
, m_mdl_semantic( code_dag->get_function_semantics( function_index))
, m_semantic( mdl_semantics_to_ext_semantics( m_mdl_semantic))
, m_name( code_dag->get_function_name( function_index))
, m_simple_name( code_dag->get_simple_function_name( function_index))
, m_db_name( get_db_name( m_name))
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported( code_dag->get_function_property(
    function_index, mi::mdl::IGenerated_code_dag::FP_IS_EXPORTED))
, m_is_uniform( code_dag->get_function_property(
    function_index, mi::mdl::IGenerated_code_dag::FP_IS_UNIFORM))
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
    const char* s = code_dag->get_cloned_function_name( function_index);
    std::string prototype_name = s ? s : "";
    m_prototype_tag = prototype_name.empty()
        ? DB::Tag() : transaction->name_to_tag( get_db_name( prototype_name).c_str());
    ASSERT( M_SCENE, m_prototype_tag || prototype_name.empty());

    const char* original_name = code_dag->get_original_function_name( function_index);
    if( original_name)
        m_original_name = original_name;

    m_tf = get_type_factory();
    m_vf = get_value_factory();
    m_ef = get_expression_factory();

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        code_dag->get_resource_tagger(),
        /*immutable*/ true,
        /*create_direct_calls*/ false,
        module_filename,
        module_name,
        m_prototype_tag,
        load_resources,
        /*user_modules_seen*/ nullptr);

    const mi::mdl::IType* return_type = code_dag->get_function_return_type( function_index);
    m_return_type = mdl_type_to_int_type( m_tf.get(), return_type);

    // function annotations
    mi::Size annotation_count = code_dag->get_function_annotation_count( function_index);
    Mdl_annotation_block annotations( annotation_count);
    for( mi::Size i = 0; i < annotation_count; ++i)
        annotations[i] = code_dag->get_function_annotation( function_index, i);
    m_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
         annotations, m_name.c_str());

    // return type annotations
    mi::Size return_annotation_count
        = code_dag->get_function_return_annotation_count( function_index);
    Mdl_annotation_block return_annotations( return_annotation_count);
    for( mi::Size i = 0; i < return_annotation_count; ++i)
        return_annotations[i] = code_dag->get_function_return_annotation( function_index, i);
    m_return_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
       return_annotations, m_name.c_str());

    // parameters/arguments
    m_defaults = m_ef->create_expression_list();
    m_parameter_annotations = m_ef->create_annotation_list();
    m_parameter_types = m_tf->create_type_list();
    m_enable_if_conditions = m_ef->create_expression_list();

    mi::Size parameter_count = code_dag->get_function_parameter_count( function_index);
    m_enable_if_users.resize( parameter_count);

    for( mi::Size i = 0; i < parameter_count; ++i) {

        const char* parameter_name = code_dag->get_function_parameter_name( function_index, i);

        // update m_parameter_types
        const mi::mdl::IType* parameter_type
            = code_dag->get_function_parameter_type( function_index, i);
        mi::base::Handle<const IType> type( mdl_type_to_int_type( m_tf.get(), parameter_type));
        m_parameter_types->add_type( parameter_name, type.get());
        m_parameter_type_names.push_back(
            code_dag->get_function_parameter_type_name( function_index, i));

        // update m_defaults
        const mi::mdl::DAG_node* default_
            = code_dag->get_function_parameter_default( function_index, i);
        if( default_) {
            mi::base::Handle<IExpression> default_int( converter.mdl_dag_node_to_int_expr(
               default_, type.get()));
            ASSERT( M_SCENE, default_int);
            m_defaults->add_expression( parameter_name, default_int.get());
        }

        // update enable_if conditions
        const mi::mdl::DAG_node* enable_if_cond
            = code_dag->get_function_parameter_enable_if_condition(function_index, i);
        if (enable_if_cond) {
            mi::base::Handle<IExpression> enable_if_cond_int(converter.mdl_dag_node_to_int_expr(
                enable_if_cond, type.get()));
            ASSERT(M_SCENE, enable_if_cond_int);
            m_enable_if_conditions->add_expression(parameter_name, enable_if_cond_int.get());
        }
        std::vector<mi::Sint32> &users = m_enable_if_users[i];
        mi::Size n_users = code_dag->get_function_parameter_enable_if_condition_users(
            function_index, i);
        for (size_t j = 0; j < n_users; ++j) {
            int param_idx = code_dag->get_function_parameter_enable_if_condition_user(
                function_index, i, int(j));
            users.push_back(param_idx);
        }

        // update m_parameter_annotations
        mi::Size parameter_annotation_count
            = code_dag->get_function_parameter_annotation_count( function_index, i);
        Mdl_annotation_block parameter_annotations( parameter_annotation_count);
        for( mi::Size j = 0; j < parameter_annotation_count; ++j)
            parameter_annotations[j]
                = code_dag->get_function_parameter_annotation( function_index, i, j);
        mi::base::Handle<IAnnotation_block> block(
            converter.mdl_dag_node_vector_to_int_annotation_block(
                parameter_annotations, m_name.c_str()));
        if( block)
            m_parameter_annotations->add_annotation_block( parameter_name, block.get());
    }

    // thumbnails: store information for on demand resolving
    m_thumbnail = (m_is_exported && module_filename && module_filename[0]) ? module_filename : "";

    // hash
    if( const mi::mdl::DAG_hash* hash = code_dag->get_function_hash( function_index))
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
    return m_name.c_str();
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

bool Mdl_function_definition::is_exported() const
{
    return m_is_exported;
}

bool Mdl_function_definition::is_uniform() const
{
    return m_is_uniform;
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

mi::Size Mdl_function_definition::get_enable_if_users(mi::Size index) const
{
    if (index < m_enable_if_users.size())
        return m_enable_if_users[index].size();
    return 0;
}

mi::Size Mdl_function_definition::get_enable_if_user(mi::Size index, mi::Size u_index) const
{
    if (index < m_enable_if_users.size()) {
        if (u_index < m_enable_if_users[index].size()) {
            return m_enable_if_users[index][u_index];
        }
    }
    return ~mi::Size(0);
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
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_function_definition( m_db_name.c_str(), m_function_ident) != 0)
        return nullptr;

    mi::Size function_index = module->get_function_definition_index( m_db_name, m_function_ident);
    ASSERT( M_SCENE, (int)function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    const mi::mdl::DAG_node* body = code_dag->get_function_body( function_index);
    if( !body)
        return nullptr;

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        code_dag->get_resource_tagger(),
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_filename*/ nullptr,
        /*module_name*/ nullptr,
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
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return 0;
    if( module->has_function_definition( m_db_name.c_str(), m_function_ident) != 0)
        return 0;

    mi::Size function_index = module->get_function_definition_index( m_db_name, m_function_ident);
    ASSERT( M_SCENE, (int)function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_function_temporary_count( function_index);
}

const IExpression* Mdl_function_definition::get_temporary(
    DB::Transaction* transaction, mi::Size index) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_function_definition( m_db_name.c_str(), m_function_ident) != 0)
        return nullptr;

    mi::Size function_index = module->get_function_definition_index( m_db_name, m_function_ident);
    ASSERT( M_SCENE, (int)function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    if( index >= code_dag->get_function_temporary_count( function_index))
        return nullptr;

    const mi::mdl::DAG_node* temporary = code_dag->get_function_temporary( function_index, index);

    Mdl_dag_converter converter(
        m_ef.get(),
        transaction,
        code_dag->get_resource_tagger(),
        /*immutable*/ true,
        /*create_direct_calls*/ true,
        /*module_filename*/ nullptr,
        /*module_name*/ nullptr,
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
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_function_definition( m_db_name.c_str(), m_function_ident) != 0)
        return nullptr;

    mi::Size function_index = module->get_function_definition_index( m_db_name, m_function_ident);
    ASSERT( M_SCENE, (int)function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    if( index >= code_dag->get_function_temporary_count( function_index))
        return nullptr;

    const char* name = code_dag->get_function_temporary_name( function_index, index);
    return *name ? name : nullptr;
}

const char* Mdl_function_definition::get_thumbnail() const
{
    if (!m_is_exported || m_thumbnail.empty() || m_thumbnail.size() < 5)
        return nullptr;

    // TODO remove .mdl/r/e encoding and the const_cast with next API and serialization change
    // to not change serialization the original module file path is stored in m_thumbnail
    // within the constructor
    size_t size = m_thumbnail.size();
    if ((m_thumbnail[size - 4] == '.' &&
         m_thumbnail[size - 3] == 'm' &&
         m_thumbnail[size - 2] == 'd' &&
         (m_thumbnail[size - 1] == 'l' || m_thumbnail[size - 1] == 'r'))
        ||
        (m_thumbnail[size - 5] == '.' &&
         m_thumbnail[size - 4] == 'm' &&
         m_thumbnail[size - 3] == 'd' &&
         m_thumbnail[size - 2] == 'l' &&
         m_thumbnail[size - 1] == 'e'))
    {
        const std::string module_filename = m_thumbnail;

        SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
        mi::base::Handle<mi::mdl::IMDL> mdl(mdlc_module->get_mdl());
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());
        m_thumbnail = DETAIL::lookup_thumbnail(
            module_filename, m_module_mdl_name, m_simple_name, m_annotations.get(),
            archive_tool.get());
    }

    if (m_thumbnail.empty())
        return nullptr;
    return m_thumbnail.c_str();
}

Mdl_function_call* Mdl_function_definition::create_function_call(
   DB::Transaction* transaction, const IExpression_list* arguments, mi::Sint32* errors) const
{
    Execution_context context;
    if (!is_valid(transaction, &context)) {
        if (errors)
            *errors = -9;
        return nullptr;
    }

    switch (m_semantic) {

    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        return create_array_constructor_call_internal(
            transaction, arguments, /*immutable=*/ false, errors);
    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
        return create_array_length_operator_call_internal(
            transaction, arguments, /*immutable=*/ false, errors);
    case mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX:
        return create_index_operator_call_internal(
            transaction, arguments, /*immutable=*/ false, errors);
    case mi::neuraylib::IFunction_definition::DS_CAST:
        return create_cast_call_internal(
            transaction, arguments, /*immutable=*/ false, errors);
    case mi::neuraylib::IFunction_definition::DS_TERNARY:
        return create_ternary_operator_call_internal(
            transaction, arguments, /*immutable=*/ false, errors);
    default:
        break;
    }

    return create_function_call_internal(
        transaction, arguments, /*allow_ek_parameter=*/ false, /*immutable=*/ false, errors);
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

    // prevent instantiation of non-exported function definitions
    if( !m_is_exported) {
        *errors = -4;
        return nullptr;
    }

    std::map<std::string, bool> needs_cast;
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // check that the provided arguments are parameters of the function definition and that their
    // types match the expected types
    if( arguments) {
        mi::Size n = arguments->get_size();
        for( mi::Size i = 0; i < n; ++i) {
            const char* name = arguments->get_name( i);
            mi::Size parameter_index = get_parameter_index(name);
            mi::base::Handle<const IType> expected_type( m_parameter_types->get_type(parameter_index));
            if( !expected_type) {
                *errors = -1;
                return nullptr;
            }
            mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
            mi::base::Handle<const IType> actual_type( argument->get_type());

            bool needs_cast_tmp = false;
            if (!argument_type_matches_parameter_type(
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
                && (kind != IExpression::EK_PARAMETER || !allow_ek_parameter)) {
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
    std::vector<mi::base::Handle<const IExpression> > context;
    for (mi::Size i = 0, n = m_parameter_types->get_size(); i < n;  ++i) {
        const char* name = get_parameter_name( i);
        mi::base::Handle<const IExpression> argument(
            arguments ? arguments->get_expression( name) : nullptr);
        if( argument) {
            // use provided argument
            mi::base::Handle<IExpression> argument_copy( m_ef->clone( argument.get(),
                transaction, /*copy_immutable_calls=*/ !immutable));
            ASSERT( M_SCENE, argument_copy);

            if (needs_cast[name]) {
                mi::base::Handle<const IType> expected_type(
                    m_parameter_types->get_type(i));
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
            argument = argument_copy;

        } else {
            // no argument provided, use default
            mi::base::Handle<const IExpression> default_( m_defaults->get_expression( name));
            if( !default_) {
                // no argument provided, no default available
                *errors = -3;
                return nullptr;
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
                deep_copy( m_ef.get(), transaction, default_.get(), context));
            ASSERT( M_SCENE, default_copy);
            argument = default_copy;
        }
        complete_arguments->add_expression( name, argument.get());
        context.push_back( argument);
    }

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        complete_arguments.get(),
        m_mdl_semantic,
        m_name.c_str(),
        m_parameter_types.get(),
        m_return_type.get(),
        immutable,
        m_enable_if_conditions.get());
    *errors = 0;
    return function_call;
}

Mdl_function_call* Mdl_function_definition::create_cast_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (!errors)
        errors = &dummy_errors;

    // check that this method is only used for the cast operator
    ASSERT(M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_CAST);

    // the  cast operator is always exported
    ASSERT(M_SCENE, m_is_exported);

    // the cast operator has no defaults
    if (!arguments) {
        *errors = -3;
        return nullptr;
    }

    // we need exactly two arguments
    mi::Size n = arguments->get_size();
    if (n != 2) {
        *errors = -3;
        return nullptr;
    }

    // check names
    mi::Size index_cast        = arguments->get_index( "cast");
    mi::Size index_cast_return = arguments->get_index( "cast_return");
    if(    index_cast        == static_cast<mi::Size>( -1)
        || index_cast_return == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return nullptr;
    }

    mi::base::Handle<const IExpression> cast_from(
        arguments->get_expression(index_cast));
    mi::base::Handle<const IType> cast_from_type(cast_from->get_type());

    mi::base::Handle<const IExpression> cast_to(
        arguments->get_expression(index_cast_return));
    mi::base::Handle<const IType> cast_to_type(cast_to->get_type());

    if (m_tf->is_compatible(cast_from_type.get(), cast_to_type.get()) < 0) {
        *errors = -2;
        return nullptr;
    }

    // the actual call only has one argument, clone it and create a new list
    mi::base::Handle<IExpression_list> new_args(m_ef->create_expression_list());
    mi::base::Handle<IExpression> new_arg(
        m_ef->clone(cast_from.get(), /*transaction*/ transaction, /*copy_immutable_calls*/ !immutable));
    new_args->add_expression("cast", new_arg.get());

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
    parameter_types->add_type("cast", cast_from_type.get());

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        new_args.get(),
        m_mdl_semantic,
        m_name.c_str(),
        parameter_types.get(),
        cast_to_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

Mdl_function_call* Mdl_function_definition::create_ternary_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (!errors)
        errors = &dummy_errors;

    // check that this method is only used for the ternary operator operator
    ASSERT(M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_TERNARY);

    // the ternary operator is always exported
    ASSERT(M_SCENE, m_is_exported);

    // the ternary operator has no defaults
    if (!arguments) {
        *errors = -3;
        return nullptr;
    }

    // we need exactly three arguments
    mi::Size n = arguments->get_size();
    if (n != 3) {
        *errors = -3;
        return nullptr;
    }

    // check names
    mi::Size index_cond      = arguments->get_index( "cond");
    mi::Size index_true_exp  = arguments->get_index( "true_exp");
    mi::Size index_false_exp = arguments->get_index( "false_exp");
    if(    index_cond      == static_cast<mi::Size>( -1)
        || index_true_exp  == static_cast<mi::Size>( -1)
        || index_false_exp == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return nullptr;
    }

    mi::base::Handle<const IExpression> cond(
        arguments->get_expression(index_cond));
    mi::base::Handle<const IType> cond_type(cond->get_type());

    mi::base::Handle<const IExpression> true_exp(
        arguments->get_expression(index_true_exp));
    mi::base::Handle<const IType> true_type(true_exp->get_type());

    mi::base::Handle<const IExpression> false_exp(
        arguments->get_expression(index_false_exp));
    mi::base::Handle<const IType> false_type(false_exp->get_type());

    if (m_tf->compare(true_type.get(), false_type.get()) != 0) {
        *errors = -2;
        return nullptr;
    }

    // the actual call has three arguments, clone them and create a new list
    mi::base::Handle<IExpression_list> new_args(m_ef->create_expression_list());
    mi::base::Handle<IExpression> new_cond(
        m_ef->clone(cond.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("cond", new_cond.get());
    mi::base::Handle<IExpression> new_true(
        m_ef->clone(true_exp.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("true_exp", new_true.get());
    mi::base::Handle<IExpression> new_false(
        m_ef->clone(false_exp.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("false_exp", new_false.get());

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
    parameter_types->add_type("cond", cond_type.get());
    parameter_types->add_type("true_exp", true_type.get());
    parameter_types->add_type("false_exp", false_type.get());

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        new_args.get(),
        m_mdl_semantic,
        m_name.c_str(),
        parameter_types.get(),
        /*return_type=*/true_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

Mdl_function_call* Mdl_function_definition::create_index_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (!errors)
        errors = &dummy_errors;

    // check that this method is only used for the index operator operator
    ASSERT(M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX);

    // the index operator is always exported
    ASSERT(M_SCENE, m_is_exported);

    // the index operator has no defaults
    if (!arguments) {
        *errors = -3;
        return nullptr;
    }

    // we need exactly two arguments
    mi::Size n = arguments->get_size();
    if (n != 2) {
        *errors = -3;
        return nullptr;
    }

    // check names
    mi::Size index_a = arguments->get_index( "a");
    mi::Size index_i = arguments->get_index( "i");
    if(    index_a == static_cast<mi::Size>( -1)
        || index_i == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return nullptr;
    }

    mi::base::Handle<const IExpression> base_expr(
        arguments->get_expression(index_a));
    mi::base::Handle<const IType> base_type(base_expr->get_type());
    mi::base::Handle<const IType> s_base_type(base_type->skip_all_type_aliases());

    mi::base::Handle<const IExpression> index_expr(
        arguments->get_expression(index_i));
    mi::base::Handle<const IType> index_type(index_expr->get_type());
    mi::base::Handle<const IType> s_index_type(index_type->skip_all_type_aliases());

    if (s_index_type->get_kind() != IType::TK_INT) {
        *errors = -2;
        return nullptr;
    }

    IType::Kind base_kind = s_base_type->get_kind();
    if (   base_kind != IType::TK_ARRAY
        && base_kind != IType::TK_VECTOR
        && base_kind != IType::TK_MATRIX) {
        *errors = -2;
        return nullptr;
    }

    mi::base::Handle<const IType_compound> s_base_type_compound(
        s_base_type->get_interface<IType_compound>());
    mi::base::Handle<const IType> ret_type( s_base_type_compound->get_component_type( 0));

    if ((base_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0 &&
        (index_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0) {
        // uniform result
        IType_factory *tf = get_type_factory();
        ret_type = mi::base::make_handle(tf->create_alias(ret_type.get(), IType::MK_UNIFORM, nullptr));
    }

    // the actual call has three arguments, clone them and create a new list
    mi::base::Handle<IExpression_list> new_args(m_ef->create_expression_list());
    mi::base::Handle<IExpression> new_base(
        m_ef->clone(base_expr.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("a", new_base.get());
    mi::base::Handle<IExpression> new_index(
        m_ef->clone(index_expr.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("i", new_index.get());

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
    parameter_types->add_type("a", base_type.get());
    parameter_types->add_type("i", index_type.get());

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        new_args.get(),
        m_mdl_semantic,
        m_name.c_str(),
        parameter_types.get(),
        ret_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

Mdl_function_call* Mdl_function_definition::create_array_length_operator_call_internal(
    DB::Transaction* transaction,
    const IExpression_list* arguments,
    bool immutable,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (!errors)
        errors = &dummy_errors;

    // check that this method is only used for the array length operator operator
    ASSERT(
        M_SCENE, m_semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH);

    // the array length operator is always exported
    ASSERT(M_SCENE, m_is_exported);

    // the array_length operator has no defaults
    if (!arguments) {
        *errors = -3;
        return nullptr;
    }

    // we need exactly one argument
    mi::Size n = arguments->get_size();
    if (n != 1) {
        *errors = -3;
        return nullptr;
    }

    // check names
    mi::Size index_a = arguments->get_index( "a");
    if( index_a == static_cast<mi::Size>( -1)) {
        *errors = -3;
        return nullptr;
    }

    mi::base::Handle<const IExpression> arr_expr(
        arguments->get_expression(index_a));
    mi::base::Handle<const IType> arr_type(arr_expr->get_type());
    mi::base::Handle<const IType> s_arr_type(arr_type->skip_all_type_aliases());

    if (s_arr_type->get_kind() != IType::TK_ARRAY) {
        *errors = -2;
        return nullptr;
    }

    IType_factory *tf = get_type_factory();
    mi::base::Handle<const IType> ret_type(tf->create_int());

    // the result is always uniform
    ret_type = mi::base::make_handle(tf->create_alias(ret_type.get(), IType::MK_UNIFORM, nullptr));

    // the actual call has one argument, clone it and create a new list
    mi::base::Handle<IExpression_list> new_args(m_ef->create_expression_list());
    mi::base::Handle<IExpression> new_arr(
        m_ef->clone(arr_expr.get(), transaction, /*copy_immutable_calls=*/!immutable));
    new_args->add_expression("a", new_arr.get());

    // create parameter type list
    mi::base::Handle<IType_list> parameter_types(m_tf->create_type_list());
    parameter_types->add_type("a", arr_type.get());

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        new_args.get(),
        m_mdl_semantic,
        m_name.c_str(),
        parameter_types.get(),
        ret_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

Mdl_function_call* Mdl_function_definition::create_array_constructor_call_internal(
   DB::Transaction* transaction,
   const IExpression_list* arguments,
   bool immutable,
   mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;

    // check that this method is only used for the array constructor
    ASSERT( M_SCENE, m_mdl_semantic == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR);

    // the array constructor is always exported
    ASSERT( M_SCENE, m_is_exported);

    // the array constructor has no defaults
    if( !arguments) {
        *errors = -3;
        return nullptr;
    }

    // the array constructor needs at least one argument
    mi::Size n = arguments->get_size();
    if( n == 0) {
        *errors = -3;
        return nullptr;
    }

    // the array constructor uses positional arguments, hence a vector is fine here
    std::vector<bool> needs_cast(n, false);
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // check that the provided arguments are all of the same type
    mi::base::Handle<const IExpression> first_argument(
        arguments->get_expression( static_cast<mi::Size>( 0)));
    mi::base::Handle<const IType> expected_type( first_argument->get_type());
    bool expected_type_uniform
        = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;

    for( mi::Size i = 1; i < n; ++i) {
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        mi::base::Handle<const IType> actual_type( argument->get_type());
        mi::Sint32 r = m_tf->is_compatible(actual_type.get(), expected_type.get());
        if (allow_cast) {
            if (r == 0) // compatible types
                needs_cast[i] = true;
            else if (r < 0) {
                *errors = -2;
                return nullptr;
            }
        }
        else {
            if (r != 1) { // different types
                *errors = -2;
                return nullptr;
            }
        }

        bool actual_type_varying
            = (actual_type->get_all_type_modifiers() & IType::MK_VARYING) != 0;
        if( actual_type_varying && expected_type_uniform) {
            *errors = -5;
            return nullptr;
        }
        IExpression::Kind kind = argument->get_kind();
        if( kind != IExpression::EK_CONSTANT &&  kind != IExpression::EK_CALL) {
            *errors = -6;
            return nullptr;
        }
        if( expected_type_uniform && return_type_is_varying( transaction, argument.get())) {
            *errors = -8;
            return nullptr;
        }
    }

    // clone arguments
    mi::base::Handle<IExpression_list> complete_arguments(
        m_ef->create_expression_list());
    for (mi::Size i = 0; i < n; ++i) {

        mi::base::Handle<const IExpression> arg0(arguments->get_expression(i));
        mi::base::Handle<IExpression> arg(
            m_ef->clone(arg0.get(), transaction, /*copy_immutable_calls=*/!immutable));
        if (needs_cast[i]) {
            arg = m_ef->create_cast(
                transaction,
                arg.get(),
                expected_type.get(),
                /*cast_db_name=*/nullptr,
                /*force_cast=*/false,
                /*direct_call=*/false,
                errors);
        }
        complete_arguments->add_expression(std::to_string( i).c_str(), arg.get());
    }

    // compute parameter types and return type
    mi::base::Handle<IType_list> parameter_types( m_tf->create_type_list());
    for( mi::Size i = 0; i < n; ++i)
        parameter_types->add_type( std::to_string( i).c_str(), expected_type.get());
    mi::base::Handle<const IType> return_type(
        m_tf->create_immediate_sized_array( expected_type.get(), n));

    Mdl_function_call* function_call = new Mdl_function_call(
        get_module(transaction),
        m_module_db_name.c_str(),
        m_function_tag,
        m_function_ident,
        complete_arguments.get(),
        m_mdl_semantic,
        m_name.c_str(),
        parameter_types.get(),
        return_type.get(),
        immutable,
        m_enable_if_conditions.get());

    *errors = 0;
    return function_call;
}

mi::mdl::IDefinition::Semantics Mdl_function_definition::get_mdl_semantic() const
{
    return m_mdl_semantic;
}

const mi::mdl::IType* Mdl_function_definition::get_mdl_return_type(
    DB::Transaction* transaction) const
{
    ASSERT( M_SCENE, m_mdl_semantic != mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR &&
        "DAG array constructor return type must be calculated");
    DB::Tag module_tag = transaction->name_to_tag(m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if (!module->is_valid(transaction, /*context=*/nullptr))
        return nullptr;
    if (module->has_function_definition(m_db_name.c_str(), m_function_ident) != 0)
        return nullptr;

    int function_index = int(module->get_function_definition_index(m_db_name, m_function_ident));
    ASSERT(M_SCENE, function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_function_return_type(function_index);
}

const mi::mdl::IType* Mdl_function_definition::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    ASSERT( M_SCENE, m_mdl_semantic != mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR &&
        "DAG array constructor parameter types cannot be retrieved, signature is (...)");
    DB::Tag module_tag = transaction->name_to_tag(m_module_db_name.c_str());
    ASSERT(M_SCENE, module_tag);

    DB::Access<Mdl_module> module(module_tag, transaction);
    if (!module->is_valid(transaction, /*context=*/nullptr))
        return nullptr;
    if (module->has_function_definition(m_db_name.c_str(), m_function_ident) != 0)
        return nullptr;

    int function_index = int(module->get_function_definition_index(m_db_name, m_function_ident));
    ASSERT(M_SCENE, function_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_function_parameter_type(function_index, index);
}

std::string Mdl_function_definition::get_mdl_name_without_parameter_types() const
{
    return m_module_mdl_name == "::<builtins>"
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

    if (module->has_function_definition(m_db_name.c_str(), m_function_ident) < 0)
        return false;

    // check defaults. is this really needed?
    for (mi::Size i = 0; i < m_defaults->get_size(); ++i) {

        mi::base::Handle<const IExpression_call> expr(
            m_defaults->get_expression<IExpression_call>(i));
        if (expr.is_valid_interface()) {
            DB::Tag call_tag = expr->get_call();
            SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
            if (class_id == ID_MDL_MATERIAL_INSTANCE) {
                DB::Access<Mdl_material_instance> minst(call_tag, transaction);
                DB::Tag_set tags_seen;
                if (!minst->is_valid(transaction, tags_seen, context))
                    return false;
            }
            else if (class_id == ID_MDL_FUNCTION_CALL) {
                DB::Access<Mdl_function_call> fcall(call_tag, transaction);
                DB::Tag_set tags_seen;
                if (!fcall->is_valid(transaction, tags_seen, context))
                    return false;
            }
        }
    }
    return true;
}

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
    if (m_annotations) {
        if (!other.m_annotations)
            return false;

        if (m_annotations->get_size() != other.m_annotations->get_size())
            return false;

        // for now, annotations must match. this implies, that all annotations
        // used by this definition that are part of the module the definition
        // comes from, also still exist.
        for (mi::Size i = 0, n = m_annotations->get_size(); i < n; ++i) {

            mi::base::Handle<const IAnnotation> anno(m_annotations->get_annotation(i));
            mi::base::Handle<const IAnnotation> other_anno(
                other.m_annotations->get_annotation(i));

            const char* anno_name = anno->get_name();
            const char* other_anno_name = other_anno->get_name();
            if (strcmp(anno_name, other_anno_name) != 0)
                return false;

            mi::base::Handle<const IExpression_list> anno_args(anno->get_arguments());
            mi::base::Handle<const IExpression_list> other_anno_args(other_anno->get_arguments());

            if (m_ef->compare(anno_args.get(), other_anno_args.get()) != 0)
                return false;
        }
    }
    else
        if (other.m_annotations)
            return false;

    return true;
}

Mdl_ident Mdl_function_definition::get_ident() const
{
    return m_function_ident;
}

void Mdl_function_definition::compute_mdl_version( const mi::mdl::IModule* mdl_module)
{
    const mi::mdl::Module* impl = mi::mdl::impl_cast<mi::mdl::Module>( mdl_module);

    if( !mdl_module->is_stdlib() && !mdl_module->is_builtins()) {
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
        m_name.c_str(), /*only_exported*/ !mdl_module->is_builtins()));
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

    serializer->write( m_module_mdl_name);
    serializer->write( m_module_db_name);
    serializer->write( m_function_tag);
    serializer->write( m_function_ident);
    serializer->write( static_cast<mi::Uint32>( m_mdl_semantic));
    serializer->write( static_cast<mi::Uint32>( m_semantic));
    serializer->write( m_name);
    serializer->write( m_simple_name);
    serializer->write( m_db_name);
    serializer->write( m_original_name);
    serializer->write( m_thumbnail);
    serializer->write( m_prototype_tag);
    serializer->write( m_is_exported);
    serializer->write( m_is_uniform);
    serializer->write( static_cast<mi::Uint32>( m_since_version));
    serializer->write( static_cast<mi::Uint32>( m_removed_version));

    m_tf->serialize_list( serializer, m_parameter_types.get());
    SERIAL::write( serializer, m_parameter_type_names);
    m_tf->serialize( serializer, m_return_type.get());
    m_ef->serialize_list( serializer, m_defaults.get());
    m_ef->serialize_annotation_block( serializer, m_annotations.get());
    m_ef->serialize_annotation_list( serializer, m_parameter_annotations.get());
    m_ef->serialize_annotation_block( serializer, m_return_annotations.get());
    m_ef->serialize_list(serializer, m_enable_if_conditions.get());

    serializer->write( m_enable_if_users);
    write( serializer, m_function_hash);

    return this + 1;
}

SERIAL::Serializable* Mdl_function_definition::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_module_mdl_name);
    deserializer->read( &m_module_db_name);
    deserializer->read( &m_function_tag);
    deserializer->read( &m_function_ident);
    mi::Uint32 semantic;
    deserializer->read( &semantic);
    m_mdl_semantic = static_cast<mi::mdl::IDefinition::Semantics>( semantic);
    deserializer->read( &semantic);
    m_semantic = static_cast<mi::neuraylib::IFunction_definition::Semantics>( semantic);
    deserializer->read( &m_name);
    deserializer->read( &m_simple_name);
    deserializer->read( &m_db_name);
    deserializer->read( &m_original_name);
    deserializer->read( &m_thumbnail);
    deserializer->read( &m_prototype_tag);
    deserializer->read( &m_is_exported);
    deserializer->read( &m_is_uniform);
    mi::Uint32 version;
    deserializer->read( &version);
    m_since_version = static_cast<mi::mdl::IMDL::MDL_version>( version);
    deserializer->read( &version);
    m_removed_version = static_cast<mi::mdl::IMDL::MDL_version>( version);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    SERIAL::read( deserializer, &m_parameter_type_names);
    m_return_type = m_tf->deserialize( deserializer);
    m_defaults = m_ef->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_parameter_annotations = m_ef->deserialize_annotation_list( deserializer);
    m_return_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    deserializer->read( &m_enable_if_users);
    read( deserializer, &m_function_hash);

    if( !m_thumbnail.empty()) {
        m_thumbnail = HAL::Ospath::convert_to_platform_specific_path( m_thumbnail);
        if( !DISK::access( m_thumbnail.c_str()))
            m_thumbnail.clear();
    }
    return this + 1;
}

void Mdl_function_definition::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;
    s << "Module MDL name: " << m_module_mdl_name << std::endl;
    s << "Module DB name: " << m_module_db_name << std::endl;
    s << "Function definition tag: " << m_function_tag.get_uint() << std::endl;
    s << "Function definition ID: " << m_function_ident << std::endl;
    s << "Function MDL semantic: " << m_mdl_semantic << std::endl;
    s << "Function semantic: " << m_semantic << std::endl;
    s << "Function definition MDL name: " << m_name << std::endl;
    s << "Function definition MDL simple name: " << m_simple_name << std::endl;
    s << "Function definition DB name: "  << m_db_name << std::endl;
    s << "Function definition MDL original name: " << m_original_name << std::endl;
    s << "Prototype tag: " << m_prototype_tag.get_uint() << std::endl;
    s << "Is exported: " << m_is_exported << std::endl;
    s << "Is uniform: " << m_is_uniform << std::endl;
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
        + dynamic_memory_consumption( m_module_mdl_name)
        + dynamic_memory_consumption( m_module_db_name)
        + dynamic_memory_consumption( m_name)
        + dynamic_memory_consumption( m_simple_name)
        + dynamic_memory_consumption( m_db_name)
        + dynamic_memory_consumption( m_original_name)
        + dynamic_memory_consumption( m_thumbnail)
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

} // namespace MDL

} // namespace MI
