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

#include "i_mdl_elements_material_definition.h"

#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_material_instance.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_utilities.h"

#include <map>
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
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/scene/i_scene_journal_types.h>


namespace MI {

namespace MDL {

Mdl_material_definition::Mdl_material_definition()
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_mdl_name()
, m_module_db_name()
, m_material_tag()
, m_material_ident( -1)
, m_name()
, m_simple_name()
, m_db_name()
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported( false)
, m_since_version( mi_mdl_IMDL_MDL_VERSION_INVALID)
, m_removed_version( mi_mdl_IMDL_MDL_VERSION_INVALID)
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
    DB::Tag material_tag,
    Mdl_ident material_ident,
    const mi::mdl::IModule* module,
    const mi::mdl::IGenerated_code_dag* code_dag,
    mi::Size material_index,
    const char* module_filename,
    const char* module_name,
    bool load_resources)
: m_tf( get_type_factory())
, m_vf( get_value_factory())
, m_ef( get_expression_factory())
, m_module_mdl_name( module_name)
, m_module_db_name( get_db_name( module_name))
, m_material_tag( material_tag)
, m_material_ident(material_ident)
, m_name( code_dag->get_material_name( material_index))
, m_simple_name( code_dag->get_simple_material_name( material_index))
, m_db_name( get_db_name( m_name))
, m_original_name()
, m_thumbnail()
, m_prototype_tag()
, m_is_exported( code_dag->get_material_exported( material_index))
, m_parameter_types()
, m_defaults()
, m_annotations()
, m_parameter_annotations()
, m_enable_if_conditions()
, m_enable_if_users()
{
    const char* s = code_dag->get_cloned_material_name( material_index);
    std::string prototype_name = s ? s : "";
    m_prototype_tag = prototype_name.empty()
        ? DB::Tag() : transaction->name_to_tag( get_db_name( prototype_name).c_str());
    ASSERT( M_SCENE, m_prototype_tag || prototype_name.empty());

    const char* original_name = code_dag->get_original_material_name( material_index);
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

    // material annotations
    mi::Size annotation_count = code_dag->get_material_annotation_count( material_index);
    Mdl_annotation_block annotations( annotation_count);
    for( mi::Size i = 0; i < annotation_count; ++i)
        annotations[i] = code_dag->get_material_annotation( material_index, i);
    m_annotations = converter.mdl_dag_node_vector_to_int_annotation_block(
        annotations, m_name.c_str());

    // parameters/arguments
    m_defaults = m_ef->create_expression_list();
    m_parameter_annotations = m_ef->create_annotation_list();
    m_parameter_types = m_tf->create_type_list();
    m_enable_if_conditions = m_ef->create_expression_list();

    mi::Size parameter_count = code_dag->get_material_parameter_count( material_index);
    m_enable_if_users.resize( parameter_count);

    for( mi::Size i = 0; i < parameter_count; ++i) {

        const char* parameter_name = code_dag->get_material_parameter_name( material_index, i);

        // update m_parameter_types
        const mi::mdl::IType* parameter_type
            = code_dag->get_material_parameter_type( material_index, i);
        mi::base::Handle<const IType> type( mdl_type_to_int_type( m_tf.get(), parameter_type));
        m_parameter_types->add_type( parameter_name, type.get());

        // update m_defaults
        const mi::mdl::DAG_node* default_
            = code_dag->get_material_parameter_default( material_index, i);
        if( default_) {
            mi::base::Handle<IExpression> default_int(converter.mdl_dag_node_to_int_expr(
                default_, type.get()));
            ASSERT( M_SCENE, default_int);
            m_defaults->add_expression( parameter_name, default_int.get());
        }

        // update enable_if conditions
        const mi::mdl::DAG_node* enable_if_cond
            = code_dag->get_material_parameter_enable_if_condition( material_index, i);
        if (enable_if_cond) {
            mi::base::Handle<IExpression> enable_if_cond_int(converter.mdl_dag_node_to_int_expr(
                enable_if_cond, type.get()));
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
        mi::Size parameter_annotation_count
            = code_dag->get_material_parameter_annotation_count( material_index, i);
        Mdl_annotation_block parameter_annotations( parameter_annotation_count);
        for( mi::Size j = 0; j < parameter_annotation_count; ++j)
            parameter_annotations[j]
                = code_dag->get_material_parameter_annotation( material_index, i, j);
        mi::base::Handle<IAnnotation_block> block(
            converter.mdl_dag_node_vector_to_int_annotation_block(
                parameter_annotations, m_name.c_str()));
        if( block)
            m_parameter_annotations->add_annotation_block( parameter_name, block.get());
    }

    // thumbnails: store information for on demand resolving
    m_thumbnail = (m_is_exported && module_filename && module_filename[0]) ? module_filename : "";

    const mi::mdl::Module* impl = mi::mdl::impl_cast<mi::mdl::Module>( module);
    m_since_version   = impl->get_mdl_version();
    m_removed_version = mi_mdl_IMDL_MDL_VERSION_INVALID;
}

DB::Tag Mdl_material_definition::get_module(DB::Transaction* transaction) const
{
    return transaction->name_to_tag( m_module_db_name.c_str());
}

const char* Mdl_material_definition::get_mdl_name() const
{
    return m_name.c_str();
}

const char* Mdl_material_definition::get_mdl_module_name() const
{
    return m_module_mdl_name.c_str();
}

const char* Mdl_material_definition::get_mdl_simple_name() const
{
    return m_simple_name.c_str();
}

void Mdl_material_definition::get_mdl_version(
    mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const
{
    since   = MDL::convert_mdl_version( m_since_version);
    removed = MDL::convert_mdl_version( m_removed_version);
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
        return nullptr;
    m_annotations->retain();
    return m_annotations.get();
}

const IAnnotation_list* Mdl_material_definition::get_parameter_annotations() const
{
    m_parameter_annotations->retain();
    return m_parameter_annotations.get();
}

const IExpression_direct_call* Mdl_material_definition::get_body( DB::Transaction* transaction) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_material_definition( m_db_name.c_str(), m_material_ident) != 0)
        return nullptr;

    mi::Size material_index = module->get_material_definition_index( m_db_name, m_material_ident);
    ASSERT( M_SCENE, (int)material_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    const mi::mdl::DAG_node* body = code_dag->get_material_value( material_index);

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
    return body_int->get_interface<const IExpression_direct_call>();
}

mi::Size Mdl_material_definition::get_temporary_count( DB::Transaction* transaction) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return 0;
    if( module->has_material_definition( m_db_name.c_str(), m_material_ident) != 0)
        return 0;

    mi::Size material_index = module->get_material_definition_index( m_db_name, m_material_ident);
    ASSERT( M_SCENE, (int)material_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_material_temporary_count( material_index);
}

const IExpression* Mdl_material_definition::get_temporary(
    DB::Transaction* transaction, mi::Size index) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_material_definition( m_db_name.c_str(), m_material_ident) != 0)
        return nullptr;

    mi::Size material_index = module->get_material_definition_index( m_db_name, m_material_ident);
    ASSERT( M_SCENE, (int)material_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    if( index >= code_dag->get_material_temporary_count( material_index))
        return nullptr;

    const mi::mdl::DAG_node* temporary = code_dag->get_material_temporary( material_index, index);

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

const char* Mdl_material_definition::get_temporary_name(
    DB::Transaction* transaction, mi::Size index) const
{
    DB::Tag module_tag = transaction->name_to_tag( m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module( module_tag, transaction);
    if( !module->is_valid( transaction, /*context=*/nullptr))
        return nullptr;
    if( module->has_material_definition( m_db_name.c_str(), m_material_ident) != 0)
        return nullptr;

    mi::Size material_index = module->get_material_definition_index( m_db_name, m_material_ident);
    ASSERT( M_SCENE, (int)material_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    if( index >= code_dag->get_material_temporary_count( material_index))
        return nullptr;

    const char* name = code_dag->get_material_temporary_name( material_index, index);
    return *name ? name : nullptr;
}

const char* Mdl_material_definition::get_thumbnail() const
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

Mdl_material_instance* Mdl_material_definition::create_material_instance(
    DB::Transaction* transaction, const IExpression_list* arguments, mi::Sint32* errors) const
{
    Execution_context context;
    if (!is_valid(transaction, &context)) {
        if (errors)
            *errors = -9;
        return nullptr;
    }
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
    if (!errors)
        errors = &dummy_errors;

    // prevent instantiation of non-exported materials
    if( !m_is_exported) {
        *errors = -4;
        return nullptr;
    }

    std::map<std::string, bool> needs_cast;
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();

    // check that the provided arguments are parameters of the material definition and that their
    // types match the expected types
    if( arguments) {
        mi::Size n = arguments->get_size();
        for( mi::Size i = 0; i < n; ++i) {
            const char* name = arguments->get_name( i);
            mi::Size parameter_index = get_parameter_index(name);
            mi::base::Handle<const IType> expected_type(m_parameter_types->get_type(parameter_index));
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
    for( mi::Size i = 0, n = m_parameter_types->get_size(); i < n; ++i) {
        const char* name = get_parameter_name( i);
        mi::base::Handle<const IExpression> argument(
            arguments ? arguments->get_expression( name) : nullptr);
        if( argument) {
            // use provided argument
            mi::base::Handle<IExpression> argument_copy( m_ef->clone(
                argument.get(), transaction, /*copy_immutable_calls=*/!immutable));
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
                // no default available
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

    Mdl_material_instance* instance = new Mdl_material_instance(
        transaction->name_to_tag(m_module_db_name.c_str()),
        m_module_db_name.c_str(),
        m_material_tag, m_material_ident, complete_arguments.get(), m_name.c_str(),
        m_parameter_types.get(),
        immutable,
        m_enable_if_conditions.get());
    *errors = 0;
    return instance;
}

const mi::mdl::IType* Mdl_material_definition::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    DB::Tag module_tag = transaction->name_to_tag(m_module_db_name.c_str());
    ASSERT( M_SCENE, module_tag);

    DB::Access<Mdl_module> module(module_tag, transaction);
    if (!module->is_valid(transaction, /*context=*/nullptr))
        return nullptr;
    if (module->has_material_definition(m_db_name.c_str(), m_material_ident) != 0)
        return nullptr;

    mi::Size material_index = module->get_material_definition_index(m_db_name, m_material_ident);
    ASSERT(M_SCENE, (int)material_index != -1);

    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());
    return code_dag->get_material_parameter_type(material_index, index);
}

const char* Mdl_material_definition::get_mdl_original_name() const
{
    return m_original_name.empty() ? nullptr : m_original_name.c_str();
}

const char* Mdl_material_definition::get_module_db_name() const
{
    ASSERT(M_SCENE, !m_module_db_name.empty());
    return m_module_db_name.c_str();
}

bool Mdl_material_definition::is_valid(
    DB::Transaction* transaction,
    Execution_context* context) const
{
    DB::Tag module_tag = get_module(transaction);
    DB::Access<Mdl_module> module(module_tag, transaction);
    if (!module->is_valid(transaction, context))
        return false;

    if (module->has_material_definition(m_db_name.c_str(), m_material_ident) < 0)
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

bool Mdl_material_definition::is_compatible(const Mdl_material_definition& other) const
{
    if (m_is_exported != other.m_is_exported)
        return false;

    if (m_prototype_tag != other.m_prototype_tag)
        return false;

    if (m_original_name != other.m_original_name)
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

Mdl_ident Mdl_material_definition::get_ident() const
{
    return m_material_ident;
}

void Mdl_material_definition::get_mdl_version(
    mi::mdl::IMDL::MDL_version& since, mi::mdl::IMDL::MDL_version& removed) const
{
    since   = m_since_version;
    removed = m_removed_version;
}

const SERIAL::Serializable* Mdl_material_definition::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serializer->write( m_module_mdl_name);
    serializer->write( m_module_db_name);
    serializer->write( m_material_tag);
    serializer->write( m_material_ident);
    serializer->write( m_name);
    serializer->write( m_simple_name);
    serializer->write( m_db_name);
    serializer->write( m_original_name);
    serializer->write( m_thumbnail);
    serializer->write( m_prototype_tag);
    serializer->write( m_is_exported);
    serializer->write( static_cast<mi::Uint32>( m_since_version));
    serializer->write( static_cast<mi::Uint32>( m_removed_version));

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

    deserializer->read( &m_module_mdl_name);
    deserializer->read( &m_module_db_name);
    deserializer->read( &m_material_tag);
    deserializer->read( &m_material_ident);
    deserializer->read( &m_name);
    deserializer->read( &m_simple_name);
    deserializer->read( &m_db_name);
    deserializer->read( &m_original_name);
    deserializer->read( &m_thumbnail);
    deserializer->read( &m_prototype_tag);
    deserializer->read( &m_is_exported);
    mi::Uint32 version;
    deserializer->read( &version);
    m_since_version = static_cast<mi::mdl::IMDL::MDL_version>( version);
    deserializer->read( &version);
    m_removed_version = static_cast<mi::mdl::IMDL::MDL_version>( version);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    m_defaults = m_ef->deserialize_list( deserializer);
    m_annotations = m_ef->deserialize_annotation_block( deserializer);
    m_parameter_annotations = m_ef->deserialize_annotation_list( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    deserializer->read( &m_enable_if_users);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);

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

    s << "Module MDL name: " << m_module_mdl_name << std::endl;
    s << "Module DB name: " << m_module_db_name << std::endl;
    s << "Material definition tag: " << m_material_tag.get_uint() << std::endl;
    s << "Material definition ID: " << m_material_ident << std::endl;
    s << "Material definition MDL name: " << m_name << std::endl;
    s << "Material definition MDL simple name: " << m_simple_name << std::endl;
    s << "Material definition DB name: " << m_db_name << std::endl;
    s << "Material definition MDL original name: " << m_original_name << std::endl;
    s << "Prototype tag: " << m_prototype_tag.get_uint() << std::endl;
    s << "Is exported: " << m_is_exported << std::endl;
    s << "Since version: " << m_since_version << std::endl;
    s << "Removed version: " << m_removed_version << std::endl;

    tmp = m_tf->dump( m_parameter_types.get());
    s << "Parameter types: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_defaults.get(), /*name*/ nullptr);
    s << "Defaults: " << tmp->get_c_str() << std::endl;
    s << "Annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_parameter_annotations.get(), /*name*/ nullptr);
    s << "Parameter annotations: " << tmp->get_c_str() << std::endl;
    tmp = m_ef->dump( transaction, m_enable_if_conditions.get(), /*name*/ nullptr);
    s << "Enable_if conditions: " << tmp->get_c_str() << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_material_definition::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_material_definition, Mdl_material_definition::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_material_definition, Mdl_material_definition::id>)
        + dynamic_memory_consumption( m_module_mdl_name)
        + dynamic_memory_consumption( m_module_db_name)
        + dynamic_memory_consumption( m_name)
        + dynamic_memory_consumption( m_simple_name)
        + dynamic_memory_consumption( m_db_name)
        + dynamic_memory_consumption( m_original_name)
        + dynamic_memory_consumption( m_thumbnail)
        + dynamic_memory_consumption( m_parameter_types)
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
    // skip m_material_tag (own tag)
    if( m_prototype_tag)
        result->insert( m_prototype_tag);
    collect_references( m_defaults.get(), result);
    collect_references( m_annotations.get(), result);
    collect_references( m_parameter_annotations.get(), result);
    collect_references( m_enable_if_conditions.get(), result);
}

} // namespace MDL

} // namespace MI
