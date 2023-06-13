/***************************************************************************************************
 * Copyright (c) 2014-2023, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the IMdl_factory implementation.
 **/

#include "pch.h"

#include "neuray_mdl_factory_impl.h"

#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/iattribute_container.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <base/data/db/i_db_transaction.h>
#include <base/lib/log/i_log_logger.h>
#include <base/system/main/access_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_module_builder.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <mdl/compiler/compilercore/compilercore_modules.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include "neuray_expression_impl.h"
#include "neuray_class_factory.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_mdl_module_builder_impl.h"
#include "neuray_mdl_module_transformer_impl.h"
#include "neuray_string_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include "neuray_scope_impl.h"

namespace MI {

namespace NEURAY {

Mdl_factory_impl::Mdl_factory_impl(
    mi::neuraylib::INeuray* neuray, const Class_factory* class_factory)
  : m_neuray( neuray),
    m_class_factory( class_factory)
{
}

Mdl_factory_impl::~Mdl_factory_impl()
{
    m_class_factory = nullptr;
    m_neuray = nullptr;
}

mi::neuraylib::IType_factory* Mdl_factory_impl::create_type_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_type_factory( transaction) : nullptr;
}

mi::neuraylib::IValue_factory* Mdl_factory_impl::create_value_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_value_factory( transaction) : nullptr;
}

mi::neuraylib::IExpression_factory* Mdl_factory_impl::create_expression_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_expression_factory( transaction) : nullptr;
}

mi::neuraylib::IMdl_execution_context* Mdl_factory_impl::create_execution_context()
{
    return new Mdl_execution_context_impl( /*internal context*/ nullptr);
}

mi::neuraylib::IMdl_execution_context* Mdl_factory_impl::clone(
    const mi::neuraylib::IMdl_execution_context* context)
{
    if( !context)
        return create_execution_context();

    MDL::Execution_context default_context;
    const MDL::Execution_context* mdl_context = unwrap_context( context, default_context);

    return new Mdl_execution_context_impl( new MDL::Execution_context( *mdl_context));
}

mi::neuraylib::IValue_texture* Mdl_factory_impl::create_texture(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    mi::neuraylib::IType_texture::Shape shape,
    mi::Float32 gamma,
    const char* selector,
    bool shared,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_context( context, default_context);

    if( !transaction) {
        add_error_message( mdl_context, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::IType_texture::Shape int_shape = ext_shape_to_int_shape( shape);
    mi::base::Handle<MDL::IValue_texture> result( MDL::create_texture(
        db_transaction, file_path, int_shape, gamma, selector, shared, mdl_context));
    // TODO MDL-1136 harmonize valid result XOR non-zero error messages count
    if( mdl_context->get_error_messages_count() > 0)
        return nullptr;
    if( !result)
        return nullptr;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_texture>( result.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_light_profile* Mdl_factory_impl::create_light_profile(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    bool shared,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_context( context, default_context);

    if( !transaction) {
        add_error_message( mdl_context, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<MDL::IValue_light_profile> result(
        MDL::create_light_profile( db_transaction, file_path, shared, mdl_context));
    if( !result)
        return nullptr;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_light_profile>( result.get(), /*owner*/ nullptr);
}

mi::neuraylib::IValue_bsdf_measurement* Mdl_factory_impl::create_bsdf_measurement(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    bool shared,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_context( context, default_context);

    if( !transaction) {
        add_error_message( mdl_context, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<MDL::IValue_bsdf_measurement> result(
        MDL::create_bsdf_measurement( db_transaction, file_path, shared, mdl_context));
    if( !result)
        return nullptr;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_bsdf_measurement>( result.get(), /*owner*/ nullptr);
}

mi::neuraylib::IMdl_module_builder* Mdl_factory_impl::create_module_builder(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::Mdl_version min_module_version,
    mi::neuraylib::Mdl_version max_module_version,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);

    if( !transaction || !module_name) {
        add_error_message( mdl_context, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    return new Mdl_module_builder_impl(
        transaction, module_name, min_module_version, max_module_version, context);
}

mi::neuraylib::IMdl_module_transformer* Mdl_factory_impl::create_module_transformer(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);

    if( !transaction || !module_name) {
        add_error_message( mdl_context, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    DB::Tag tag = db_transaction->name_to_tag( module_name);
    if( !tag || db_transaction->get_class_id( tag) != MDL::Mdl_module::id) {
        add_error_message( mdl_context, "Invalid module name.", -2);
        return nullptr;
    }

    DB::Access<MDL::Mdl_module> db_module( tag, db_transaction);
    mi::base::Handle<const mi::mdl::IModule> mdl_module( db_module->get_mdl_module());
    const mi::mdl::Module* mdl_module_impl = mi::mdl::impl_cast<mi::mdl::Module>( mdl_module.get());
    if( mdl_module_impl->is_compiler_owned() || mdl_module_impl->is_native()) {
        add_error_message( mdl_context, "Builtin modules cannot be transformed.", -3);
        return nullptr;
    }

    return new Mdl_module_transformer_impl( transaction, module_name, mdl_module.get());
}

const mi::IString* Mdl_factory_impl::get_db_module_name( const char* mdl_name)
{
    if( !mdl_name)
        return nullptr;

    bool mdle = MDL::is_mdle( mdl_name);
    if( !mdle && !MDL::starts_with_scope( mdl_name))
        return nullptr;

    std::string normalized_mdl_name = MDL::get_mdl_name_from_load_module_arg( mdl_name, mdle);
    std::string result = MDL::get_db_name( normalized_mdl_name);
    return new String_impl( result.c_str());
}

const mi::IString* Mdl_factory_impl::get_db_definition_name( const char* mdl_name)
{
    if( !mdl_name)
        return nullptr;

    std::string mdl_name_str = mdl_name;
    if( !MDL::starts_with_scope( mdl_name_str))
        return nullptr;

    std::string result = MDL::get_db_name( mdl_name);
    return new String_impl( result.c_str());
}

void Mdl_factory_impl::analyze_uniform(
    mi::neuraylib::ITransaction* transaction,
    const char* root_name,
    bool root_uniform,
    const mi::neuraylib::IExpression* query_expr,
    bool& query_result,
    mi::IString* error_path,
    mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    query_result = false;
    if( error_path)
        error_path->set_c_str( "");

    if( !transaction || !root_name) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( root_name);
    if( !tag) {
        add_error_message( context_impl, "Invalid root name.", -2);
        return;
    }

    mi::base::Handle<MDL::IExpression_direct_call> int_root_expr;
    SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
    if( class_id != MDL::Mdl_function_call::id) {
        add_error_message( context_impl, "Invalid root type.", -3);
        return;
    }

    DB::Access<MDL::Mdl_function_call> fc( tag, db_transaction);
    mi::base::Handle<const MDL::IExpression_list> arguments_int( fc->get_arguments());
    DB::Tag def_tag = fc->get_function_definition( db_transaction);
    DB::Access<MDL::Mdl_function_definition> def( def_tag, db_transaction);
    int_root_expr = def->create_direct_call( db_transaction, arguments_int.get(), nullptr);
    if( !int_root_expr) {
        add_error_message( context_impl, "Failed to create root expression.", -4);
        return;
    }

    mi::base::Handle<const MDL::IType> int_root_type( int_root_expr->get_type());
    int_root_type = MDL::get_type_factory()->create_alias(
        int_root_type.get(), root_uniform, /*symbol*/ nullptr);

    mi::base::Handle<const MDL::IExpression> int_query_expr(
        get_internal_expression( query_expr));

    // If query_expr is an argument to fc, we need to take the cloning during create_direct_call()
    // into account.
    if( int_query_expr) {
        for( mi::Size i = 0, n = arguments_int->get_size(); i < n; ++i) {
            mi::base::Handle<const MDL::IExpression> argument_int(
                arguments_int->get_expression( i));
            if( int_query_expr != argument_int.get())
                continue;
            mi::base::Handle<const MDL::IExpression_list> cloned_arguments_int(
                int_root_expr->get_arguments());
            int_query_expr = cloned_arguments_int->get_expression( i);
        }
    }

    std::vector<bool> dummy_result_parameters;
    std::string int_error_path;
    MDL::Mdl_module_builder::analyze_uniform(
        db_transaction,
        int_root_expr.get(),
        root_uniform,
        int_query_expr.get(),
        dummy_result_parameters,
        query_result,
        int_error_path,
        context_impl);
    if( error_path)
        error_path->set_c_str( int_error_path.c_str());
}

const mi::IString* Mdl_factory_impl::decode_name( const char* name)
{
    if( !name)
        return nullptr;

    std::string result = MDL::decode( name, /*strict*/ false);
    if( result.empty() && strlen( name) > 0)
        return nullptr;

    return new String_impl( result.c_str());
}

const mi::IString* Mdl_factory_impl::encode_module_name( const char* name)
{
    if( !name)
        return nullptr;

    std::string result = MDL::encode_module_name( name);
    return new String_impl( result.c_str());
}

const mi::IString* Mdl_factory_impl::encode_function_definition_name(
    const char* name, const mi::IArray* parameter_types) const
{
    if( !name)
        return nullptr;

    mi::Size n = parameter_types ? parameter_types->get_length() : 0;
    std::vector<std::string> parameter_types_int;

    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const mi::IString> parameter_type(
            parameter_types->get_value<mi::IString>( i));
        if( !parameter_type)
            return nullptr;
        parameter_types_int.push_back( parameter_type->get_c_str());
    }

    std::string result = MDL::encode_name_plus_signature( name, parameter_types_int);
    return new String_impl( result.c_str());
}

const mi::IString* Mdl_factory_impl::encode_type_name( const char* name) const
{
    if( !name)
        return nullptr;

    std::string result = MDL::encode_name_without_signature( name);
    return new String_impl( result.c_str());
}

bool Mdl_factory_impl::is_valid_mdl_identifier( const char* name) const
{
    return m_mdl->is_valid_mdl_identifier( name);
}

mi::Sint32 Mdl_factory_impl::start()
{
    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    m_mdl = mdlc_module->get_mdl();
    return 0;
}

mi::Sint32 Mdl_factory_impl::shutdown()
{
    m_mdl.reset();
    return 0;
}

} // namespace NEURAY

} // namespace MI

