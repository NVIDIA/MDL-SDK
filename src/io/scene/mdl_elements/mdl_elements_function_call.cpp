/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_function_call.h"

#include "i_mdl_elements_compiled_material.h"
#include "i_mdl_elements_expression.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_module.h"
#include "i_mdl_elements_type.h"
#include "i_mdl_elements_utilities.h"
#include "i_mdl_elements_value.h"
#include "mdl_elements_utilities.h"
#include "mdl_elements_expression.h"

#include <sstream>

#include <boost/core/ignore_unused.hpp>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/config/config.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_transaction_wrapper.h>
#include <base/data/db/i_db_info.h>
#include <base/data/serial/i_serializer.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

namespace MI {

namespace MDL {

using mi::mdl::as;

Mdl_function_call::Mdl_function_call()
  : m_tf( get_type_factory()),
    m_vf( get_value_factory()),
    m_ef( get_expression_factory()),
    m_definition_ident(),
    m_mdl_semantic( mi::mdl::IDefinition::DS_UNKNOWN),
    m_immutable( false), // avoid ubsan warning with swap() and temporaries
    m_is_material( false)
{
}

namespace {

const char* check_non_null( const char* s)
{
    ASSERT( M_SCENE, s && "string argument should be non-NULL");
    return s;
}

}

Mdl_function_call::Mdl_function_call(
    DB::Tag module_tag,
    const char* module_db_name,
    DB::Tag definition_tag,
    Mdl_ident definition_ident,
    bool is_material,
    IExpression_list* arguments,
    mi::mdl::IDefinition::Semantics semantic,
    const char* definition_name,
    const IType_list* parameter_types,
    const IType* return_type,
    bool immutable,
    const IExpression_list* enable_if_conditions)
  : m_tf( get_type_factory()),
    m_vf( get_value_factory()),
    m_ef( get_expression_factory()),
    m_module_tag( module_tag),
    m_definition_tag( definition_tag),
    m_definition_ident( definition_ident),
    m_mdl_semantic( semantic),
    m_module_db_name( check_non_null( module_db_name)),
    m_definition_name( check_non_null( definition_name)),
    m_definition_db_name( get_db_name( m_definition_name)),
    m_immutable( immutable),
    m_is_material( is_material),
    m_parameter_types( parameter_types, mi::base::DUP_INTERFACE),
    m_return_type( return_type, mi::base::DUP_INTERFACE),
    m_arguments( arguments, mi::base::DUP_INTERFACE),
    m_enable_if_conditions( enable_if_conditions, mi::base::DUP_INTERFACE)
{
}

Mdl_function_call::Mdl_function_call( const Mdl_function_call& other)
  : SCENE::Scene_element<Mdl_function_call, ID_MDL_FUNCTION_CALL>( other),
    m_tf( other.m_tf),
    m_vf( other.m_vf),
    m_ef( other.m_ef),
    m_module_tag( other.m_module_tag),
    m_definition_tag( other.m_definition_tag),
    m_definition_ident( other.m_definition_ident),
    m_mdl_semantic( other.m_mdl_semantic),
    m_module_db_name( other.m_module_db_name),
    m_definition_name( other.m_definition_name),
    m_definition_db_name( other.m_definition_db_name),
    m_immutable( other.m_immutable),
    m_is_material( other.m_is_material),
    m_parameter_types( other.m_parameter_types),
    m_return_type( other.m_return_type),
    m_enable_if_conditions( other.m_enable_if_conditions) // shared, no clone necessary
{
    // Clone only the expression list itself for performance reasons. Arguments are never modified
    // in place, only by setting new expression.
    const Expression_list* other_arguments_impl
        = static_cast<const Expression_list*>( other.m_arguments.get());
    m_arguments = new Expression_list( *other_arguments_impl);
}

DB::Tag Mdl_function_call::get_function_definition( DB::Transaction* transaction) const
{
    if( !is_valid_definition( transaction))
        return DB::Tag();

    return  m_definition_tag;
}

const char* Mdl_function_call::get_mdl_function_definition() const
{
    return m_definition_name.c_str();
}

const IType* Mdl_function_call::get_return_type() const
{
    m_return_type->retain();
    return m_return_type.get();
}

mi::Size Mdl_function_call::get_parameter_count() const
{
    return m_arguments->get_size();
}

const char* Mdl_function_call::get_parameter_name( mi::Size index) const
{
    return m_arguments->get_name( index);
}

mi::Size Mdl_function_call::get_parameter_index( const char* name) const
{
    return m_arguments->get_index( name);
}

const IType_list* Mdl_function_call::get_parameter_types() const
{
    m_parameter_types->retain();
    return m_parameter_types.get();
}

const IExpression_list* Mdl_function_call::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

const IExpression_list* Mdl_function_call::get_enable_if_conditions() const
{
    m_enable_if_conditions->retain();
    return m_enable_if_conditions.get();
}

bool Mdl_function_call::is_valid(
    DB::Transaction* transaction, Execution_context* context) const
{
    DB::Tag_set tags_seen;
    return is_valid(transaction, tags_seen, context);
}

bool Mdl_function_call::is_valid(
    DB::Transaction* transaction, DB::Tag_set& tags_seen, Execution_context* context) const
{
    DB::Tag module_tag = get_module( transaction);
    DB::Access<Mdl_module> module(module_tag, transaction);
    if (!module->is_valid(transaction, context))
        return false;

    if (module->has_definition(m_is_material,m_definition_db_name, m_definition_ident) != 0) {
        add_error_message(
            context, "The function definition '" + m_definition_db_name + "' "
            "does no longer exist or has interface changes.", -1);
        return false;
    }

    for (mi::Size i = 0, n = m_arguments->get_size(); i < n; ++i) {
        mi::base::Handle<const IExpression> arg(m_arguments->get_expression(i));
        if (arg->get_kind() == IExpression::EK_CALL) {
            mi::base::Handle<const IExpression_call> arg_call(
                arg->get_interface<IExpression_call>());
            DB::Tag call_tag = arg_call->get_call();
            if (!call_tag)
                continue;
            if (!tags_seen.insert(call_tag).second)
                return false; // cycle in graph, always invalid.
            SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
            if (class_id != ID_MDL_FUNCTION_CALL) {
                add_error_message(
                    context, "The function call attached to parameter '"
                    + std::string(m_arguments->get_name(i)) + "' has a wrong element type.", -1);
                return false;
            }
            DB::Access<Mdl_function_call> fcall(call_tag, transaction);
            if (!fcall->is_valid(transaction, tags_seen, context)) {
                add_error_message(
                    context, "The function call attached to parameter '"
                    + std::string(m_arguments->get_name(i)) + "' is invalid.", -1);
                return false;
            }
            tags_seen.erase(call_tag);
        }
    }
    return true;
}

mi::Sint32 Mdl_function_call::repair(
    DB::Transaction* transaction,
    bool repair_invalid_calls,
    bool remove_invalid_calls,
    mi::Uint32 level,
    Execution_context* context)
{
    if (m_immutable) // immutable calls cannot be changed.
        return -3;

    ASSERT(M_SCENE, m_module_tag);
    DB::Access<Mdl_module> module(m_module_tag, transaction);
    // cannot restore if we refer to an invalid module
    if (!module->is_valid(transaction, context))
        return -1;

    mi::Sint32 ret
        = module->has_definition(m_is_material, m_definition_db_name, m_definition_ident);
    if (ret == -1) {
        // a definition of that name does no longer exist
        m_definition_tag = DB::Tag();
        m_definition_ident = -1;
        add_error_message(
            context, "The definition '" + m_definition_db_name +
            "' does no longer exist in the module.", -1);
        return -1;
    }
    else if (ret == -2) {

        if (level == 0 || repair_invalid_calls) {

            mi::Size index = module->get_definition_index(
                m_is_material, m_definition_db_name, Mdl_ident(-1));
            ASSERT(M_SCENE, index != mi::Size(-1));
            DB::Tag def_tag = module->get_definition(m_is_material, index);
            DB::Access<Mdl_function_definition> fdef(def_tag, transaction);

            if( fdef->is_material()) {

            // compare argument types with new parameter types
            if (m_arguments->get_size() != fdef->get_parameter_count()) {
                // for now, we cannot adapt to this.
                m_definition_tag = DB::Tag();
                m_definition_ident = -1;
                add_error_message(
                    context, "The parameter count of definition '" + m_definition_db_name +
                    "' has changed.", -1);
                return -1;
            }

            mi::base::Handle<const IType_list> param_types(fdef->get_parameter_types());
            for (mi::Size i = 0, n = fdef->get_parameter_count(); i < n; ++i) {

                const char* param_name = fdef->get_parameter_name(i);
                const char* arg_name = m_arguments->get_name(i);
                if (strcmp(param_name, arg_name) == 0) {

                    mi::base::Handle<const IType> ptype(param_types->get_type(i));
                    mi::base::Handle<const IExpression> arg(m_arguments->get_expression(i));
                    mi::base::Handle<const IType> atype(arg->get_type());

                    bool needs_cast;
                    if (!argument_type_matches_parameter_type(
                        m_tf.get(),
                        atype.get(),
                        ptype.get(),
                        /*allow_cast=*/false,
                        needs_cast)) {

                        m_definition_tag = DB::Tag();
                        m_definition_ident = -1;
                        return -1;
                    }
                }
                else {
                    m_definition_tag = DB::Tag();
                    m_definition_ident = -1;
                    return -1;
                }
            }

            } else {

            // in this case, parameter types must match (part of signature)
            // does the return type match, too?
            mi::base::Handle<const IType> def_ret_type(fdef->get_return_type());

            bool needs_cast;
            if (!argument_type_matches_parameter_type(
                m_tf.get(),
                def_ret_type.get(),
                m_return_type.get(),
                /*allow_cast=*/false,
                needs_cast)) {

                // different type, cannot promote
                m_definition_tag = DB::Tag();
                m_definition_ident = -1;
                add_error_message(
                    context, "The return type of definition '" + m_definition_db_name +
                    "' has changed.", -1);
                return -1;
            }

            }
            m_definition_tag = def_tag;
            m_definition_ident = fdef->get_ident();
        }
        else
            return -1;
    }

    // try to fix invalid arguments
    DB::Access<Mdl_function_definition> fct_def(m_definition_tag, transaction);
    mi::base::Handle<const IExpression_list> defaults(fct_def->get_defaults());

    if(repair_arguments(
        transaction,
        m_arguments.get(), defaults.get(),
        repair_invalid_calls, remove_invalid_calls, ++level, context) != 0) {

        m_definition_tag = DB::Tag();
        m_definition_ident = -1;
        return -1;
    }
    return 0;
}

mi::Sint32 Mdl_function_call::set_arguments(
    DB::Transaction* transaction, const IExpression_list* arguments)
{
    if( !arguments)
        return -1;
    mi::Size n = arguments->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        const char* name = arguments->get_name( i);
        mi::base::Handle<const IExpression> argument( arguments->get_expression(name));
        mi::Sint32 result = set_argument( transaction, name, argument.get());
        if( result != 0)
            return result;
    }

    return 0;
}

mi::Sint32 Mdl_function_call::set_argument(
    DB::Transaction* transaction, mi::Size index, const IExpression* argument)
{
    if( !argument)
        return -1;

    mi::base::Handle<const IType> expected_type( m_parameter_types->get_type( index));
    if( !expected_type)
        return -2;

    mi::base::Handle<const IType> actual_type( argument->get_type());

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    bool allow_cast = mdlc_module->get_implicit_cast_enabled();
    bool needs_cast = false;
    if( !argument_type_matches_parameter_type(
        m_tf.get(),
        actual_type.get(),
        expected_type.get(),
        allow_cast,
        needs_cast))
        return -3;

    if( m_immutable)
        return -4;

    bool actual_type_varying   = (actual_type->get_all_type_modifiers()   & IType::MK_VARYING) != 0;
    bool expected_type_uniform = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
    if( actual_type_varying && expected_type_uniform)
        return -5;

    IExpression::Kind kind = argument->get_kind();
    if( kind != IExpression::EK_CONSTANT && kind != IExpression::EK_CALL)
        return -6;

    if( expected_type_uniform && return_type_is_varying( transaction, argument))
        return -8;

    mi::base::Handle<IExpression> argument_copy(
        m_ef->clone( argument, transaction, /*copy_immutable_calls=*/ true));
    if( !argument_copy)
        return -6;

    if( needs_cast) {
        mi::Sint32 errors = 0;
        argument_copy = m_ef->create_cast(
            transaction,
            argument_copy.get(),
            expected_type.get(),
            /*db_element_name*/nullptr,
            /*force_cast=*/false,
            /*direct_call=*/false,
            &errors);
        ASSERT( M_SCENE, argument_copy); // should always succeed.
    }

    m_arguments->set_expression( index, argument_copy.get());
    return 0;
}

mi::Sint32 Mdl_function_call::set_argument(
    DB::Transaction* transaction, const char* name, const IExpression* argument)
{
    if( !name || !argument)
        return -1;

    mi::Size index = get_parameter_index( name);
    return set_argument( transaction, index, argument);
}

mi::Sint32 Mdl_function_call::reset_argument(
    DB::Transaction* transaction, mi::Size index)
{
    mi::base::Handle<const IType> expected_type( m_parameter_types->get_type( index));
    if( !expected_type)
        return -2;

    if( m_immutable)
        return -4;

    if( !is_valid_definition( transaction))
        return -9;

    DB::Access<Mdl_function_definition> definition( m_definition_tag, transaction);
    mi::base::Handle<const IExpression_list> defaults( definition->get_defaults());
    const char* name = get_parameter_name( index);

    // consider default first
    mi::base::Handle<const IExpression> argument( defaults->get_expression( name));
    if( argument) {

        // clone the default (also resolves parameter references)
        std::vector<mi::base::Handle<const IExpression> > call_context;
        for( mi::Size i = 0; i < index; ++i)
            call_context.push_back( make_handle( m_arguments->get_expression( i)));
        argument = deep_copy( m_ef.get(), transaction, argument.get(), call_context);
        ASSERT( M_SCENE, argument);

        // check for uniform violation
        bool expected_type_uniform
            = (expected_type->get_all_type_modifiers() & IType::MK_UNIFORM) != 0;
        if( expected_type_uniform && return_type_is_varying( transaction, argument.get()))
            argument = 0;
    }

    // otherwise consider range annotations or use default-constructed value
    if( !argument) {
         mi::base::Handle<const IAnnotation_list> annotation_list(
             definition->get_parameter_annotations());
         mi::base::Handle<const IAnnotation_block>  annotation_block(
             annotation_list->get_annotation_block( name));
         mi::base::Handle<IValue> value(
             m_vf->create( expected_type.get(), annotation_block.get()));
         ASSERT( M_SCENE, value);
         argument = m_ef->create_constant( value.get());
    }

    m_arguments->set_expression( index, argument.get());
    return 0;
}

mi::Sint32 Mdl_function_call::reset_argument(
    DB::Transaction* transaction, const char* name)
{
    if( !name)
        return -1;

    mi::Size index = get_parameter_index( name);
    return reset_argument( transaction, index);
}

void Mdl_function_call::make_mutable( DB::Transaction* transaction)
{
    if( !is_valid_definition( transaction))
        return;

    // Immutable calls do not store the module tag if it points to the same module to avoid cyclic
    // references.
    ASSERT( M_SCENE, m_immutable || m_module_tag);
    if( !m_module_tag)
        m_module_tag = get_module( transaction);
    m_immutable = false;
    ASSERT( M_SCENE, m_immutable || m_module_tag);
}

mi::mdl::IDefinition::Semantics Mdl_function_call::get_mdl_semantic() const
{
    return m_mdl_semantic;
}

const mi::mdl::IType* Mdl_function_call::get_mdl_return_type( DB::Transaction* transaction) const
{
    if( !is_valid_definition( transaction))
        return nullptr;

    // For template-like functions the MDL return type has to be calculated from the internal
    // return type (of the call, not the definition). DS_INTRINSIC_DAG_ARRAY_LENGTH is
    // template-like, but the return type is fixed.
    mi::neuraylib::IFunction_definition::Semantics semantic
        = mdl_semantics_to_ext_semantics( m_mdl_semantic);
    if(    semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        || semantic == mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX
        || semantic == mi::neuraylib::IFunction_definition::DS_CAST
        || semantic == mi::neuraylib::IFunction_definition::DS_TERNARY) {

        DB::Tag module_tag = get_module( transaction);
        DB::Access<Mdl_module> module( module_tag, transaction);
        mi::base::Handle<const mi::mdl::IModule> mdl_module( module->get_mdl_module());
        mi::mdl::IType_factory* tf = mdl_module->get_type_factory();
        return int_type_to_mdl_type( m_return_type.get(), *tf);
    }

    DB::Access<Mdl_function_definition> definition( m_definition_tag, transaction);
    return definition->get_mdl_return_type( transaction);
}

const mi::mdl::IType* Mdl_function_call::get_mdl_parameter_type(
    DB::Transaction* transaction, mi::Uint32 index) const
{
    if( !is_valid_definition( transaction))
        return nullptr;

    // For template-like functions the MDL parameter type has to be calculated from the internal
    // parameter type (of the call, not the definition).
    mi::neuraylib::IFunction_definition::Semantics semantic
        = mdl_semantics_to_ext_semantics( m_mdl_semantic);
    if(     semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
        ||  semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DAG_ARRAY_LENGTH
        || (semantic == mi::neuraylib::IFunction_definition::DS_ARRAY_INDEX && index == 0)
        ||  semantic == mi::neuraylib::IFunction_definition::DS_CAST
        || (semantic == mi::neuraylib::IFunction_definition::DS_TERNARY && index > 0)) {

        mi::base::Handle<const IType> parameter_type( m_parameter_types->get_type( index));
        DB::Tag module_tag = get_module( transaction);
        DB::Access<Mdl_module> module( module_tag, transaction);
        mi::base::Handle<const mi::mdl::IModule> mdl_module( module->get_mdl_module());
        mi::mdl::IType_factory* tf = mdl_module->get_type_factory();
        return int_type_to_mdl_type( parameter_type.get(), *tf);
    }

    DB::Access<Mdl_function_definition> definition( m_definition_tag, transaction);
    return definition->get_mdl_parameter_type( transaction, index);
}

DB::Tag Mdl_function_call::get_module( DB::Transaction* transaction) const
{
    // Immutable calls do not store the module tag if it points to the same module to avoid cyclic
    // references.
    ASSERT( M_SCENE, m_immutable || m_module_tag);
    return m_module_tag ? m_module_tag : transaction->name_to_tag( m_module_db_name.c_str());
}

bool Mdl_function_call::is_valid_definition( DB::Transaction* transaction) const
{
    DB::Access<Mdl_module> module( get_module( transaction), transaction);
    return module->has_definition( m_is_material, m_definition_db_name, m_definition_ident) == 0;
}

void Mdl_function_call::swap( Mdl_function_call& other)
{
    SCENE::Scene_element<Mdl_function_call, Mdl_function_call::id>::swap( other);

    std::swap( m_module_tag, other.m_module_tag);
    std::swap( m_definition_tag, other.m_definition_tag);
    std::swap( m_definition_ident, other.m_definition_ident);
    std::swap( m_mdl_semantic, other.m_mdl_semantic);
    m_module_db_name.swap( other.m_module_db_name);
    m_definition_name.swap( other.m_definition_name);
    m_definition_db_name.swap( other.m_definition_db_name);

    std::swap( m_immutable, other.m_immutable);
    std::swap( m_is_material, other.m_is_material);
    std::swap( m_parameter_types, other.m_parameter_types);
    std::swap( m_return_type, other.m_return_type);
    std::swap( m_arguments, other.m_arguments);
    std::swap( m_enable_if_conditions, other.m_enable_if_conditions);
}

mi::mdl::IGenerated_code_lambda_function* Mdl_function_call::create_jitted_function(
    DB::Transaction* transaction,
    bool environment_context,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max,
    Sint32* errors) const
{
    ASSERT( M_SCENE, !m_is_material);

    mi::Sint32 dummy_errors;
    if( !errors)
        errors = &dummy_errors;
    *errors = -1;

    Execution_context context;
    if (!is_valid(transaction, &context)) {
        *errors = -5;
        return nullptr;
    }

    // get the JIT code generator
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::ICode_generator_jit> generator_jit
        = mi::base::make_handle( mdl->load_code_generator( "jit"))
            .get_interface<mi::mdl::ICode_generator_jit>();
    if( !generator_jit.is_valid_interface()) {
        *errors = -1;
        return nullptr;
    }

    // get function definition and check its return type
    DB::Access<Mdl_function_definition> function_definition( m_definition_tag, transaction);
    const mi::mdl::IType* return_type = function_definition->get_mdl_return_type( transaction);
    bool is_ok = false;
    if( return_type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_COLOR) {
        is_ok = true;
    } else if( const mi::mdl::IType_struct* s_type = as<mi::mdl::IType_struct>( return_type)) {
        // check for a type struct equal to ::base::texture_return
        if( s_type->get_field_count() == 2) {
            const mi::mdl::IType*   c_type;
            const mi::mdl::IType*   f_type;
            const mi::mdl::ISymbol* dummy;
            s_type->get_field( 0, c_type, dummy);
            s_type->get_field( 1, f_type, dummy);

            if( c_type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_COLOR &&
                f_type->skip_type_alias()->get_kind() == mi::mdl::IType::TK_FLOAT) {
                is_ok = true;
            }
        }
    }
    if( !is_ok) {
        *errors = -2;
        return nullptr;
    }

    // get code DAG
    ASSERT(M_SCENE, m_module_tag);
    DB::Access<Mdl_module> module(m_module_tag, transaction);
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());

    // create a lambda function for an environment function
    mi::base::Handle<mi::mdl::ILambda_function> lambda_func(
        mdl->create_lambda_function( environment_context
            ?  mi::mdl::ILambda_function::LEC_ENVIRONMENT
            : mi::mdl::ILambda_function::LEC_DISPLACEMENT ));

    // set JIT generator parameters
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    const CONFIG::Config_registry& registry = config_module->get_configuration();
    mi::mdl::Options& options = generator_jit->access_options();
    options.set_option( MDL_JIT_OPTION_ENABLE_RO_SEGMENT, "true");
    options.set_option( MDL_JIT_OPTION_USE_BITANGENT, "true");
    int jit_opt_level = 0;
    if( registry.get_value( "jit_opt_level", jit_opt_level)) {
        options.set_option( MDL_JIT_OPTION_OPT_LEVEL, std::to_string( jit_opt_level).c_str());
    }
    bool jit_fast_math = false;
    if( registry.get_value( "jit_fast_math", jit_fast_math)) {
        options.set_option( MDL_JIT_OPTION_FAST_MATH, jit_fast_math ? "true" : "false");
    }

    int function_index = int(module->get_definition_index( m_is_material,
        m_definition_db_name, m_definition_ident));
    if (function_index == -1) {
        *errors = -5;
        return nullptr;
    }

    // convert m_arguments to DAG nodes
    Mdl_dag_builder<mi::mdl::IDag_builder> builder(
        transaction, lambda_func.get(), /*compiled_material*/ nullptr);
    mi::Size n_params = code_dag->get_function_parameter_count(function_index);
    std::vector<mi::mdl::DAG_call::Call_argument> mdl_arguments( n_params);

    for( mi::Size i = 0; i < n_params; ++i) {
        const char* parameter_name = code_dag->get_function_parameter_name(function_index, i);
        const mi::mdl::IType* parameter_type
            = code_dag->get_function_parameter_type(function_index, i);
        mi::base::Handle<const IExpression> argument( m_arguments->get_expression( parameter_name));
        const mi::mdl::DAG_node* arg
            = builder.int_expr_to_mdl_dag_node( parameter_type, argument.get());
        if( !arg) {
            LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                "Type mismatch, call of an unsuitable DB element, or cycle in a graph rooted "
                "at the function definition \"%s\".",
                m_definition_db_name.c_str());
            *errors = -3;
            return nullptr;
        }
        mdl_arguments[i].arg        = arg;
        mdl_arguments[i].param_name = parameter_name;
    }

    // create DAG node for the environment function
    const mi::mdl::DAG_node* call = lambda_func->create_call(
        m_definition_name.c_str(), function_definition->get_mdl_semantic(),
        mdl_arguments.data(), int( n_params), return_type);

    // if return type is a struct type assume it is ::base::texture_return (see above) and wrap
    // DAG node by a select to extract the tint field
    if( const mi::mdl::IType_struct* s_type = as<mi::mdl::IType_struct>( return_type)) {
        const mi::mdl::IType*   f_type;
        const mi::mdl::ISymbol* f_name;

        s_type->get_field( 0, f_type, f_name);

        std::string name( s_type->get_symbol()->get_name());
        name += '.';
        name += f_name->get_name();
        name += '(';
        name += s_type->get_symbol()->get_name();
        name += ')';

        mi::mdl::DAG_call::Call_argument args[1];

        args[0].arg        = call;
        args[0].param_name = "s";
        call = lambda_func->create_call(
            name.c_str(), mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS, args, 1, f_type);
    }

    if( !environment_context) {
        size_t idx = lambda_func->store_root_expr( call);
        ASSERT( M_SCENE, idx == 0);
        boost::ignore_unused( idx);
    } else {
        lambda_func->set_body( call);
    }

    // compile the environment lambda function
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(transaction, mdlc_module->get_module_wait_queue(), {});
    Mdl_call_resolver resolver( transaction);
    mi::mdl::IGenerated_code_lambda_function* jitted_func;
    if( !environment_context ) {
        jitted_func = generator_jit->compile_into_switch_function(
            lambda_func.get(), &module_cache, &resolver, /*ctx=*/nullptr, 1, 0);
    } else {
        jitted_func = generator_jit->compile_into_environment(
            lambda_func.get(), &module_cache, &resolver, /*ctx=*/nullptr);
    }
    if( !jitted_func) {
        *errors = -4;
        return nullptr;
    }

    *errors = 0;
    return jitted_func;
}

/// This transaction wrapper caches results of various methods on DB::Transaction.
///
/// The cache assumes that the transaction's view on the database does *not* change during the
/// caches lifetime. Do *not* use the cache if this not guaranteed.
///
/// Cached methods: get_class_id(), name_to_tag(), get_element().
class Caching_transaction : public DB::Transaction_wrapper
{
public:
    Caching_transaction( DB::Transaction* transaction)
      : DB::Transaction_wrapper( transaction) { }

    ~Caching_transaction()
    {
        for( auto &entry : m_elements)
            entry.second->unpin();
    }

    SERIAL::Class_id get_class_id( DB::Tag tag) final
    {
        auto it = m_class_ids.find( tag);
        if( it != m_class_ids.end())
            return it->second;

        SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
        m_class_ids[tag] = class_id;
        return class_id;
    }

    // Invalid tags are not cached. Unclear if relevant.
    DB::Tag name_to_tag( const char* name) final
    {
        auto it = m_tags.find( name);
        if( it != m_tags.end())
            return it->second;

        DB::Tag tag = m_transaction->name_to_tag( name);
        if( tag)
            m_tags[name] = tag;
        return tag;
    }

    DB::Info* get_element( DB::Tag tag, bool do_wait) final
    {
        auto it = m_elements.find( tag);
        if( it != m_elements.end()) {
            it->second->pin();
            return it->second;
        }

        DB::Info* element = m_transaction->get_element( tag, do_wait);
        element->pin();
        m_elements[tag] = element;
        return element;
    }

private:
    // Allow comparisons of C-style string with std::strings to avoid temporaries when searching in
    // a map.
    struct String_less
    {
        using is_transparent = void;
        bool operator()( const std::string& lhs, const char* rhs) const
        { return strcmp( lhs.c_str(), rhs) < 0; }
        bool operator()( const char* lhs, const std::string& rhs) const
        { return strcmp( lhs, rhs.c_str()) < 0; }
        bool operator()( const std::string& lhs, const std::string& rhs) const
        { return lhs < rhs; }
    };

    std::map<DB::Tag, SERIAL::Class_id>         m_class_ids;
    std::map<std::string, DB::Tag, String_less> m_tags;
    std::map<DB::Tag, DB::Info*>                m_elements;
};

Mdl_compiled_material* Mdl_function_call::create_compiled_material(
    DB::Transaction* transaction_non_cached,
    bool class_compilation,
    Execution_context* context) const
{
    Caching_transaction transaction_cached(transaction_non_cached);
    DB::Transaction* transaction = &transaction_cached;

    ASSERT( M_SCENE, m_is_material);

    if( !is_valid( transaction, context)) {
        add_error_message( context, "The material instance is invalid.", -1);
        return nullptr;
    }

    mi::base::Handle<const mi::mdl::IGenerated_code_dag::IMaterial_instance> instance(
        create_dag_material_instance(
            transaction,
            /*use_temporaries*/ true,
            class_compilation,
            context));
    if( !instance.is_valid_interface())
        return nullptr;

    ASSERT(M_SCENE, m_module_tag);
    DB::Access<Mdl_module> module( m_module_tag, transaction);
    const char* module_filename = module->get_filename();
    const char* module_name = module->get_mdl_name();

    mi::Float32 mdl_meters_per_scene_unit
        = context->get_option<mi::Float32>( MDL_CTX_OPTION_METERS_PER_SCENE_UNIT);
    mi::Float32 mdl_wavelength_min
        = context->get_option<mi::Float32>( MDL_CTX_OPTION_WAVELENGTH_MIN);
    mi::Float32 mdl_wavelength_max
        = context->get_option<mi::Float32>( MDL_CTX_OPTION_WAVELENGTH_MAX);
    bool resolve_resources = context->get_option<bool>( MDL_CTX_OPTION_RESOLVE_RESOURCES);

    return new Mdl_compiled_material(
        transaction, instance.get(), module_filename, module_name,
        mdl_meters_per_scene_unit, mdl_wavelength_min, mdl_wavelength_max, resolve_resources);
}

const mi::mdl::IGenerated_code_dag::IMaterial_instance*
Mdl_function_call::create_dag_material_instance(
    DB::Transaction* transaction,
    bool use_temporaries,
    bool class_compilation,
    Execution_context* context) const
{
    ASSERT( M_SCENE, m_is_material);
    ASSERT( M_SCENE, context);

    // get code DAG
    DB::Access<Mdl_module> module( m_module_tag, transaction);
    mi::base::Handle<const mi::mdl::IGenerated_code_dag> code_dag( module->get_code_dag());

    // create new MDL material instance
    mi::mdl::IGenerated_code_dag::Error_code error_code;

    int material_index = int(module->get_definition_index(
        m_is_material, m_definition_db_name, m_definition_ident));
    if (material_index == -1)
        return nullptr;

    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> instance(
        code_dag->create_material_instance(material_index, &error_code));
    ASSERT( M_SCENE, error_code == 0);
    ASSERT( M_SCENE, instance.is_valid_interface());

    bool fold_meters_per_scene_unit = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_METERS_PER_SCENE_UNIT);
    mi::Float32 mdl_meters_per_scene_unit = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_METERS_PER_SCENE_UNIT);
    mi::Float32 mdl_wavelength_min = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_WAVELENGTH_MIN);
    mi::Float32 mdl_wavelength_max = context->get_option<mi::Float32>(
        MDL_CTX_OPTION_WAVELENGTH_MAX);
    bool fold_ternary_on_df = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_TERNARY_ON_DF);
    bool remove_dead_parameters = context->get_option<bool>(
        MDL_CTX_OPTION_REMOVE_DEAD_PARAMETERS);
    bool fold_all_bool_parameters = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_ALL_BOOL_PARAMETERS);
    bool fold_all_enum_parameters = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_ALL_ENUM_PARAMETERS);
    mi::base::Handle<const mi::IArray> fold_parameters(
        context->get_interface_option<const mi::IArray>( MDL_CTX_OPTION_FOLD_PARAMETERS));
    bool fold_trivial_cutout_opacity = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_TRIVIAL_CUTOUT_OPACITY);
    bool fold_transparent_layers = context->get_option<bool>(
        MDL_CTX_OPTION_FOLD_TRANSPARENT_LAYERS);
    bool ignore_noinline = context->get_option<bool>(MDL_CTX_OPTION_IGNORE_NOINLINE);

    // convert fold_parameters to array of const char*
    std::vector<const char*> fold_parameters_converted;
    mi::Size n_fold = fold_parameters ? fold_parameters->get_length() : 0;
    for( size_t i = 0; i < n_fold; ++i) {

        mi::base::Handle<const mi::IString> element( fold_parameters->get_element<mi::IString>( i));
        if( !element) {
            add_error_message( context,
                "An element in the array for the context option \"fold_parameters\" does not have "
                "the type mi::IString.", -1);
            return nullptr;
        }

        fold_parameters_converted.push_back( element->get_c_str());
    }

    // convert m_arguments to DAG nodes
    Mdl_dag_builder<mi::mdl::IDag_builder> builder(
        transaction, instance.get(), /*compiled_material*/ nullptr);
    mi::Size n = code_dag->get_material_parameter_count( material_index);
    std::vector<const mi::mdl::DAG_node*> mdl_arguments( n);

    for( mi::Size i = 0; i < n; ++i) {
        const mi::mdl::IType* parameter_type
            = code_dag->get_material_parameter_type( material_index, i);
        mi::base::Handle<const IExpression> argument( m_arguments->get_expression( i));
        mdl_arguments[i] = builder.int_expr_to_mdl_dag_node( parameter_type, argument.get());
        if( !mdl_arguments[i]) {
            add_error_message( context,
                "Type mismatch, call of an unsuitable DB element, or call cycle in a graph rooted "
                "at the material definition \"" + m_definition_db_name + "\".", -1);
            return nullptr;
        }
    }

    bool resolve_resources = context->get_option<bool>(MDL_CTX_OPTION_RESOLVE_RESOURCES);

    // initialize MDL material instance
    Call_evaluator<mi::mdl::IGenerated_code_dag> call_evaluator(
        code_dag.get(), transaction, resolve_resources);
    Mdl_call_resolver resolver(transaction);

    mi::Uint32 flags = 0;
    if( class_compilation) {
        flags = mi::mdl::IGenerated_code_dag::IMaterial_instance::CLASS_COMPILATION
              | mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_RESOURCE_SHARING
              | mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_ARGUMENT_INLINE;
        if( fold_ternary_on_df)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_TERNARY_ON_DF;
        if ( remove_dead_parameters)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_DEAD_PARAMS;
        if( fold_all_bool_parameters)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_BOOL_PARAMS;
        if( fold_all_enum_parameters)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_ENUM_PARAMS;
        if( fold_trivial_cutout_opacity)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_TRIVIAL_CUTOUT_OPACITY;
        if( fold_transparent_layers)
            flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::NO_TRANSPARENT_LAYERS;
    } else {
        flags =  mi::mdl::IGenerated_code_dag::IMaterial_instance::INSTANCE_COMPILATION;
    }
    if (ignore_noinline)
        flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::IGNORE_NOINLINE;
    if (context->get_option<bool>(MDL_CTX_OPTION_TARGET_MATERIAL_MODEL_MODE))
        flags |= mi::mdl::IGenerated_code_dag::IMaterial_instance::TARGET_MATERIAL_MODEL;

    error_code = instance->initialize(
        &resolver,
        /*resource_modifier=*/ nullptr,
        code_dag.get(),
        n,
        mdl_arguments.data(),
        use_temporaries,
        flags,
        class_compilation ? nullptr : &call_evaluator,
        fold_meters_per_scene_unit,
        mdl_meters_per_scene_unit,
        mdl_wavelength_min,
        mdl_wavelength_max,
        fold_parameters_converted.data(),
        fold_parameters_converted.size());

    switch( error_code) {

        case mi::mdl::IGenerated_code_dag::EC_NONE:
            break;

        case mi::mdl::IGenerated_code_dag::EC_ARGUMENT_TYPE_MISMATCH: {
            add_message(
                context,
                Message(
                    mi::base::MESSAGE_SEVERITY_ERROR,
                    "Type mismatch for an argument in a graph rooted at the material "
                    "definition \"" + m_definition_db_name + "\".",
                    mi::mdl::IGenerated_code_dag::EC_ARGUMENT_TYPE_MISMATCH,
                    Message::MSG_COMPILER_DAG),
                -1);
            return nullptr;
        }

        case mi::mdl::IGenerated_code_dag::EC_WRONG_TRANSMISSION_ON_THIN_WALLED: {
            // Warn if we detect different transmission
            add_message(
                context,
                Message(
                    mi::base::MESSAGE_SEVERITY_WARNING,
                    "The thin-walled material instance rooted of the material definition \""
                    + m_definition_db_name
                    + "\" has different transmission for surface and backface.",
                    mi::mdl::IGenerated_code_dag::EC_WRONG_TRANSMISSION_ON_THIN_WALLED,
                    Message::MSG_COMPILER_DAG),
                0);
            break;
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
    convert_and_log_messages( msgs, context);

    if( msgs.get_error_message_count() > 0) {
        context->set_result( -3);
        return nullptr;
    }

    instance->retain();
    return instance.get();
}

const SERIAL::Serializable* Mdl_function_call::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, m_module_tag);
    SERIAL::write( serializer, m_definition_tag);
    SERIAL::write( serializer, m_definition_ident);
    SERIAL::write_enum( serializer, m_mdl_semantic);
    SERIAL::write( serializer, m_module_db_name);
    SERIAL::write( serializer, m_definition_name);
    SERIAL::write( serializer, m_definition_db_name);
    SERIAL::write( serializer, m_immutable);
    SERIAL::write( serializer, m_is_material);

    m_tf->serialize_list( serializer, m_parameter_types.get());
    m_tf->serialize( serializer, m_return_type.get());
    m_ef->serialize_list( serializer, m_arguments.get());
    m_ef->serialize_list( serializer, m_enable_if_conditions.get());

    return this + 1;
}

SERIAL::Serializable* Mdl_function_call::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    SERIAL::read( deserializer, &m_module_tag);
    SERIAL::read( deserializer, &m_definition_tag);
    SERIAL::read( deserializer, &m_definition_ident);
    SERIAL::read_enum( deserializer, &m_mdl_semantic);
    SERIAL::read( deserializer, &m_module_db_name);
    SERIAL::read( deserializer, &m_definition_name);
    SERIAL::read( deserializer, &m_definition_db_name);
    SERIAL::read( deserializer, &m_immutable);
    SERIAL::read( deserializer, &m_is_material);

    m_parameter_types = m_tf->deserialize_list( deserializer);
    m_return_type = m_tf->deserialize( deserializer);
    m_arguments = m_ef->deserialize_list( deserializer);
    m_enable_if_conditions = m_ef->deserialize_list( deserializer);

    return this + 1;
}

void Mdl_function_call::dump( DB::Transaction* transaction) const
{
    std::ostringstream s;
    s << std::boolalpha;
    mi::base::Handle<const mi::IString> tmp;

    s << "Module tag: " << m_module_tag.get_uint() << std::endl;
    s << "Module DB name: \"" << m_module_db_name << '\"' << std::endl;
    s << "Function definition tag: " << m_definition_tag.get_uint() << std::endl;
    s << "Function definition identifier: " << m_definition_ident << std::endl;
    s << "Function definition MDL name: \"" << m_definition_name << '\"' << std::endl;
    s << "Function definition DB name: \"" << m_definition_db_name << '\"' << std::endl;
    tmp = m_ef->dump( transaction, m_arguments.get(), /*name*/ nullptr);
    s << "Arguments: " << tmp->get_c_str() << std::endl;
    s << "Immutable: " << m_immutable << std::endl;
    s << "Is material: " << m_is_material << std::endl;
    tmp = m_ef->dump( transaction, m_enable_if_conditions.get(), /*name*/ nullptr);
    s << "Enable_if conditions: " << tmp->get_c_str() << std::endl;

    s << std::endl;
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_function_call::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_function_call, Mdl_function_call::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_function_call, Mdl_function_call::id>)
        + dynamic_memory_consumption( m_module_db_name)
        + dynamic_memory_consumption( m_definition_name)
        + dynamic_memory_consumption( m_definition_db_name)
        + dynamic_memory_consumption( m_parameter_types)
        + dynamic_memory_consumption( m_return_type)
        + dynamic_memory_consumption( m_arguments)             // might be shared
        + dynamic_memory_consumption( m_enable_if_conditions); // usually shared
}

DB::Journal_type Mdl_function_call::get_journal_flags() const
{
    return SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE;
}

Uint Mdl_function_call::bundle( DB::Tag* results, Uint size) const
{
    return 0;
}

void Mdl_function_call::get_scene_element_references( DB::Tag_set* result) const
{
    // Immutable calls do not store the module tag if it points to the same module to avoid cyclic
    // references.
    ASSERT( M_SCENE, m_immutable || m_module_tag);
    if( m_module_tag)
        result->insert( m_module_tag);

    if( m_definition_tag)
        result->insert( m_definition_tag);

    collect_references( m_arguments.get(), result);
    collect_references( m_enable_if_conditions.get(), result);
}

mi::Sint32 Mdl_function_call::repair_call(
    DB::Transaction* transaction,
    const DB::Access<Mdl_function_call>& call_access,
    mi::base::Handle<IExpression>& new_expr,
    const IExpression *default_expr,
    const IExpression_call* arg_call,
    IExpression_factory* ef,
    IValue_factory* vf,
    bool repair_invalid_calls,
    bool remove_invalid_calls,
    mi::Uint32 level,
    Execution_context* context
)
{
    DB::Edit<Mdl_function_call> call_edit(call_access);
    mi::Sint32 res = call_edit->repair(
        transaction, repair_invalid_calls, remove_invalid_calls, level, context);
    if (res == 0)
        return 0;

    if (remove_invalid_calls) {

        if (default_expr) {
            new_expr =
                ef->clone(default_expr, transaction, /*copy_immutable_calls=*/true);
        }
        else { // create a value
            mi::base::Handle<const IType> arg_type(arg_call->get_type());
            mi::base::Handle<IValue> new_val(vf->create(arg_type.get()));
            new_expr = ef->create_constant(new_val.get());
        }
        return 0;
    }
    return -1;
}

mi::Sint32 Mdl_function_call::repair_arguments(
    DB::Transaction* transaction,
    IExpression_list* arguments,
    const IExpression_list* defaults,
    bool repair_invalid_calls,
    bool remove_invalid_calls,
    mi::Uint32 level,
    Execution_context* context)
{
    mi::base::Handle<IExpression_factory> ef(get_expression_factory());
    mi::base::Handle<IValue_factory> vf(get_value_factory());

    for (mi::Size i = 0, n = arguments->get_size(); i < n; ++i) {

        mi::base::Handle<const IExpression_call> arg_call(
            arguments->get_expression<IExpression_call>(i));
        if (arg_call.is_valid_interface()) {
            DB::Tag call_tag = arg_call->get_call();
            if (!call_tag.is_valid())
                continue;

            SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
            if (class_id != ID_MDL_FUNCTION_CALL)
                continue;

            const char* arg_name = arguments->get_name(i);
            mi::base::Handle<const IExpression> default_expr(
                defaults->get_expression(defaults->get_index(arg_name)));

            mi::base::Handle<IExpression> new_expr;
            DB::Access<Mdl_function_call> fcall(call_tag, transaction);
            if (!fcall->is_valid(transaction, context)) {
                if (repair_call(
                    transaction,
                    fcall,
                    new_expr,
                    default_expr.get(),
                    arg_call.get(),
                    ef.get(),
                    vf.get(),
                    repair_invalid_calls,
                    remove_invalid_calls,
                    level,
                    context) != 0) {
                        add_error_message( context,
                            STRING::formatted_string(
                                "The call \"%s\" attached to argument \"%s\" could not be "
                                "repaired.", transaction->tag_to_name(call_tag), arg_name), -1);
                    return -1;
                }
            }
            if (new_expr)
                arguments->set_expression(i, new_expr.get());
        }
    }
    return 0;
}

} // namespace MDL

} // namespace MI
