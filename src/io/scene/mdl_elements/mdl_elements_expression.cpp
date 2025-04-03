/***************************************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief      Source for the IExpression hierarchy and IExpression_factory implementation.

#include "pch.h"

#include "i_mdl_elements_module.h"
#include "i_mdl_elements_annotation_definition_proxy.h"
#include "i_mdl_elements_function_call.h"
#include "i_mdl_elements_function_definition.h"
#include "i_mdl_elements_utilities.h"

#include "mdl_elements_expression.h"
#include "mdl_elements_detail.h"
#include "mdl_elements_type.h"
#include "mdl_elements_utilities.h"
#include "mdl_elements_value.h"

#include <cstring>
#include <sstream>

#include <mi/neuraylib/istring.h>
// for mi::neuraylib::IExpression_factory::Comparison_options
#include <mi/neuraylib/iexpression.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_access.h>
#include <base/data/serial/i_serializer.h>

namespace MI {

namespace MDL {

mi::Sint32 Expression_constant::set_value( IValue* value)
{
    if( !value)
        return -1;

    mi::base::Handle<const IType> type( value->get_type());
    if( Type_factory::compare_static( m_type.get(), type.get()) != 0)
        return -2;

    m_type = type;
    m_value = make_handle_dup( value);
    return 0;
}

mi::Size Expression_constant::get_memory_consumption() const
{
    // m_type is shared with m_value
    return sizeof( *this)
        + dynamic_memory_consumption( m_value);
}

mi::Size Expression_call::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

mi::Size Expression_parameter::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

DB::Tag Expression_direct_call::get_definition( DB::Transaction* transaction) const
{
    if( !transaction)
        return m_definition_ident.first;

    DB::Access<Mdl_module> module( m_module_tag, transaction);
    SERIAL::Class_id class_id = transaction->get_class_id( m_definition_ident.first);
    if( class_id != ID_MDL_FUNCTION_DEFINITION)
        return {};

    if( module->has_definition(
        /*is_material*/ true, m_definition_db_name, m_definition_ident.second) == 0)
        return m_definition_ident.first;

    if( module->has_definition(
        /*is_material*/ false, m_definition_db_name, m_definition_ident.second) == 0)
        return m_definition_ident.first;

    return {};
}

const IExpression_list* Expression_direct_call::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

mi::Size Expression_direct_call::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type)
        + dynamic_memory_consumption( m_arguments);
}

mi::Size Expression_temporary::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_type);
}

Expression_list::Expression_list( mi::Size initial_capacity)
{
    m_index_name.reserve( initial_capacity);
    m_expressions.reserve( initial_capacity);
}

mi::Size Expression_list::get_size() const
{
    return m_expressions.size();
}

mi::Size Expression_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);

    // For typical list sizes a linear search is much faster than maintaining a map from names to
    // indices.
    for( mi::Size i = 0; i < m_index_name.size(); ++i)
        if( m_index_name[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Expression_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return nullptr;
    return m_index_name[index].c_str();
}

const IExpression* Expression_list::get_expression( mi::Size index) const
{
    if( index >= m_expressions.size())
        return nullptr;
    m_expressions[index]->retain();
    return m_expressions[index].get();
}

const IExpression* Expression_list::get_expression( const char* name) const
{
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return nullptr;
    return get_expression( index);
}

mi::Sint32 Expression_list::set_expression( mi::Size index, const IExpression* expression)
{
    if( !expression)
        return -1;
    if( index >= m_expressions.size())
        return -2;
    m_expressions[index] = make_handle_dup( expression);
    return 0;
}

mi::Sint32 Expression_list::set_expression( const char* name, const IExpression* expression)
{
    if( !expression)
        return -1;
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -2;
    m_expressions[index] = make_handle_dup( expression);
    return 0;
}

mi::Sint32 Expression_list::add_expression( const char* name, const IExpression* expression)
{
    if( !name || !expression)
        return -1;
    mi::Size index = get_index( name);
    if( index != static_cast<mi::Size>( -1))
        return -2;
    m_expressions.push_back( make_handle_dup( expression));
    m_index_name.push_back( name);
    return 0;
}

void Expression_list::add_expression_unchecked(
    const char* name, const IExpression* expression)
{
    m_expressions.push_back( make_handle_dup( expression));
    m_index_name.push_back( name);
}

mi::Size Expression_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_expressions);
}

const char* Annotation_definition::get_mdl_parameter_type_name(mi::Size index) const
{
    if( index >= m_parameter_type_names.size())
        return nullptr;
    return m_parameter_type_names[index].c_str();
}

mi::neuraylib::IAnnotation_definition::Semantics Annotation_definition::get_semantic() const
{
    return m_semantic;
}

bool Annotation_definition::is_exported() const
{
    return m_is_exported;
}

void Annotation_definition::get_mdl_version(
    mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const
{
    since   = m_since_version;
    removed = m_removed_version;
}

mi::Size Annotation_definition::get_parameter_count() const
{
    return m_parameter_types->get_size();
}

const char* Annotation_definition::get_parameter_name( mi::Size index) const
{
    return m_parameter_types->get_name( index);
}

mi::Size Annotation_definition::get_parameter_index( const char* name) const
{
    return m_parameter_types->get_index( name);
}

const IType_list* Annotation_definition::get_parameter_types() const
{
    m_parameter_types->retain();
    return m_parameter_types.get();
}

const IExpression_list* Annotation_definition::get_defaults() const
{
    m_parameter_defaults->retain();
    return m_parameter_defaults.get();
}

const IAnnotation_block* Annotation_definition::get_annotations() const
{
    if (!m_annotations)
        return nullptr;

    m_annotations->retain();
    return m_annotations.get();
}

const IAnnotation* Annotation_definition::create_annotation(const IExpression_list* arguments) const
{
    // check that all provided arguments exist in the definition
    if (arguments) {
        for (mi::Size i = 0, n = arguments->get_size(); i < n; ++i) {
            const char* name = arguments->get_name(i);
            if (m_parameter_types->get_index(name) == mi::Size(-1))
                return nullptr;
        }
    }

    // build up complete arguments
    mi::base::Handle<IExpression_factory> ef(get_expression_factory());
    mi::base::Handle<IType_factory> tf(get_type_factory());
    mi::Size n = m_parameter_types->get_size();
    mi::base::Handle<IExpression_list> complete_arguments(ef->create_expression_list(n));
    for (mi::Size i = 0; i < n; ++i) {

        const char* name = m_parameter_types->get_name(i);
        mi::base::Handle<const IExpression> arg(
            arguments ? arguments->get_expression(name) : nullptr);
        if (arg) {
            if (arg->get_kind() != IExpression::EK_CONSTANT)
                return nullptr;
            mi::base::Handle<const IType> arg_type(arg->get_type());
            mi::base::Handle<const IType> param_type(m_parameter_types->get_type(i));
            bool needs_cast = false;
            if (!argument_type_matches_parameter_type(
                tf.get(),
                arg_type.get(),
                param_type.get(),
                /*allow_cast*/ false,
                needs_cast)) {
                return nullptr;
            }
        }
        else {
            arg = m_parameter_defaults->get_expression(name);
            if (!arg) // no default
                return nullptr;
        }
        mi::base::Handle<IExpression> cloned_arg(ef->clone(
            arg.get(),
            /*transaction*/ nullptr,
            /*copy_immutable_calls*/ false));
        complete_arguments->add_expression(name, cloned_arg.get());
    }
    return new Annotation(m_name.c_str(), complete_arguments.get());
}

mi::Size Annotation_definition::get_memory_consumption() const
{
    return sizeof(*this)
        + dynamic_memory_consumption(m_name)
        + dynamic_memory_consumption(m_module_mdl_name)
        + dynamic_memory_consumption(m_module_db_name)
        + dynamic_memory_consumption(m_simple_name)
        + dynamic_memory_consumption(m_parameter_type_names)
        + dynamic_memory_consumption(m_parameter_types)
        + dynamic_memory_consumption(m_parameter_defaults)
        + dynamic_memory_consumption(m_annotations);
}

std::string Annotation_definition::get_mdl_name_without_parameter_types() const
{
    return m_module_mdl_name + "::" + m_simple_name;
}

const IExpression_list* Annotation::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

const IAnnotation_definition* Annotation::get_definition( DB::Transaction* transaction) const
{
    const std::string& db_name = get_db_name_annotation_definition( m_name);
    DB::Tag definition_proxy_tag = transaction->name_to_tag( db_name.c_str());
    if( !definition_proxy_tag)
        return nullptr;

    DB::Access<Mdl_annotation_definition_proxy> definition_proxy(
        definition_proxy_tag, transaction);
    std::string module_db_name = definition_proxy->get_db_module_name();

    DB::Tag module_tag = transaction->name_to_tag( module_db_name.c_str());
    ASSERT( M_SCENE, module_tag.is_valid());

    DB::Access<Mdl_module> module( module_tag, transaction);
    return module->get_annotation_definition( m_name.c_str());
}

mi::Size Annotation::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_name)
        + dynamic_memory_consumption( m_arguments);
}

Annotation_block::Annotation_block( mi::Size initial_capacity)
{
    m_annotations.reserve( initial_capacity);
}

mi::Size Annotation_block::get_size() const
{
    return m_annotations.size();
}

const IAnnotation* Annotation_block::get_annotation( mi::Size index) const
{
    if( index >= m_annotations.size())
        return nullptr;
    m_annotations[index]->retain();
    return m_annotations[index].get();
}

mi::Sint32 Annotation_block::set_annotation( mi::Size index, const IAnnotation* annotation)
{
    if( !annotation)
        return -1;
    if( index >= m_annotations.size())
        return -2;
    m_annotations[index] = make_handle_dup( annotation);
    return 0;
}

mi::Sint32 Annotation_block::add_annotation( const IAnnotation* annotation)
{
    if( !annotation)
        return -1;
    m_annotations.push_back( make_handle_dup( annotation));
    return 0;
}

mi::Size Annotation_block::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_annotations);
}

Annotation_list::Annotation_list( mi::Size initial_capacity)
{
    m_index_name.reserve( initial_capacity);
    m_annotation_blocks.reserve( initial_capacity);
}

mi::Size Annotation_list::get_size() const
{
    return m_annotation_blocks.size();
}

mi::Size Annotation_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);

    // For typical list sizes a linear search is much faster than maintaining a map from names to
    // indices.
    for( mi::Size i = 0; i < m_index_name.size(); ++i)
        if( m_index_name[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Annotation_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return nullptr;
    return m_index_name[index].c_str();
}

const IAnnotation_block* Annotation_list::get_annotation_block( mi::Size index) const
{
    if( index >= m_annotation_blocks.size())
        return nullptr;
    m_annotation_blocks[index]->retain();
    return m_annotation_blocks[index].get();
}

const IAnnotation_block* Annotation_list::get_annotation_block( const char* name) const
{
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return nullptr;
    return get_annotation_block( index);
}

mi::Sint32 Annotation_list::set_annotation_block(
    mi::Size index, const IAnnotation_block* block)
{
    if( !block)
        return -1;
    if( index >= m_annotation_blocks.size())
        return -2;
    m_annotation_blocks[index] = make_handle_dup( block);
    return 0;
}

mi::Sint32 Annotation_list::set_annotation_block(
    const char* name, const IAnnotation_block* block)
{
    if( !block)
        return -1;
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -2;
    m_annotation_blocks[index] = make_handle_dup( block);
    return 0;
}

mi::Sint32 Annotation_list::add_annotation_block(
    const char* name, const IAnnotation_block* block)
{
    if( !name || !block)
        return -1;
    mi::Size index = get_index( name);
    if( index != static_cast<mi::Size>( -1))
        return -2;
    m_annotation_blocks.push_back( make_handle_dup( block));
    m_index_name.push_back( name);
    return 0;
}

void Annotation_list::add_annotation_block_unchecked(
    const char* name, const IAnnotation_block* block)
{
    m_annotation_blocks.push_back( make_handle_dup( block));
    m_index_name.push_back( name);
}

mi::Size Annotation_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_annotation_blocks);
}

Expression_factory::Expression_factory( IValue_factory* value_factory)
  : m_value_factory( value_factory, mi::base::DUP_INTERFACE)
{
}

IValue_factory* Expression_factory::get_value_factory() const
{
    m_value_factory->retain();
    return m_value_factory.get();
}

IExpression_constant* Expression_factory::create_constant( IValue* value) const
{
    return value ? new Expression_constant( value) : nullptr;
}

const IExpression_constant* Expression_factory::create_constant( const IValue* value) const
{
    // The const_cast is safe here since we return a constant expression which does not give
    // access to a mutable value.
    return value ? new Expression_constant( const_cast<IValue*>( value)) : nullptr;
}

IExpression_call* Expression_factory::create_call( const IType* type, DB::Tag tag) const
{
    return type && tag ? new Expression_call( type, tag) : nullptr;
}

IExpression_parameter* Expression_factory::create_parameter(
    const IType* type, mi::Size index) const
{
    return type ? new Expression_parameter( type, index) : nullptr;
}

IExpression_direct_call* Expression_factory::create_direct_call(
    const IType* type,
    DB::Tag module_tag,
    const Mdl_tag_ident& definition_ident,
    const char* definition_db_name,
    IExpression_list* arguments) const
{
    if( !type || !definition_ident.first || !definition_db_name || !arguments)
        return nullptr;

    return new Expression_direct_call(
        type, module_tag, definition_ident, definition_db_name, arguments);
}

IExpression_temporary* Expression_factory::create_temporary(
    const IType* type, mi::Size index) const
{
    return type ? new Expression_temporary( type, index) : nullptr;
}

IAnnotation* Expression_factory::create_annotation(
    DB::Transaction* transaction, const char* name, const IExpression_list* arguments) const
{
    if( !name || !arguments)
        return nullptr;

    mi::Size n = arguments->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        if( argument->get_kind() != IExpression::EK_CONSTANT)
            return nullptr;
    }

    // No annotation present during deserialization, skip sanity check then.
    if( transaction) {
        const std::string& db_name = get_db_name_annotation_definition( name);
        DB::Tag definition_proxy_tag = transaction->name_to_tag( db_name.c_str());
        if( !definition_proxy_tag)
            return nullptr;
    }

    return new Annotation( name, arguments);
}

IAnnotation_definition* Expression_factory::create_annotation_definition(
    const char* name,
    const char* module_name,
    const char* simple_name,
    const std::vector<std::string>& parameter_type_names,
    mi::neuraylib::IAnnotation_definition::Semantics sema,
    bool is_exported,
    mi::neuraylib::Mdl_version since_version,
    mi::neuraylib::Mdl_version removed_version,
    const IType_list* parameter_types,
    const IExpression_list* parameter_defaults,
    const IAnnotation_block* annotations) const
{
    if (!name || !parameter_types)
        return nullptr;

    return new Annotation_definition(
        name, module_name, simple_name, parameter_type_names, sema, is_exported,
        since_version, removed_version, parameter_types, parameter_defaults, annotations);
}

IExpression_list* Expression_factory::create_expression_list( mi::Size initial_capacity) const
{
    return new Expression_list( initial_capacity);
}

IAnnotation_block* Expression_factory::create_annotation_block( mi::Size initial_capacity) const
{
    return new Annotation_block( initial_capacity);
}

IAnnotation_list* Expression_factory::create_annotation_list( mi::Size initial_capacity) const
{
    return new Annotation_list( initial_capacity);
}

IAnnotation_definition_list* Expression_factory::create_annotation_definition_list(
    mi::Size initial_capacity) const
{
    return new Annotation_definition_list( initial_capacity);
}

IExpression* Expression_factory::clone(
    const IExpression* expr,
    DB::Transaction* transaction,
    bool copy_immutable_calls) const
{
    if( !expr)
        return nullptr;

    IExpression::Kind kind = expr->get_kind();

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            mi::base::Handle<IValue> clone_value( m_value_factory->clone( value.get()));
            return create_constant( clone_value.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            if( copy_immutable_calls) {
                ASSERT( M_SCENE, transaction);
                DB::Tag call_tag = expr_call->get_call();
                SERIAL::Class_id class_id = transaction->get_class_id( call_tag);
                if( class_id != ID_MDL_FUNCTION_CALL)
                    return nullptr;
                DB::Access<Mdl_function_call> fc( call_tag, transaction);
                if( fc->is_immutable())
                    return deep_copy( this, transaction, expr_call.get(), /*call_context*/ nullptr);
            }
            return create_call( type.get(), expr_call->get_call());
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            return create_parameter( type.get(), expr_parameter->get_index());
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            mi::base::Handle<IExpression_list> clone_arguments(
                clone( arguments.get(), transaction, copy_immutable_calls));
            Mdl_tag_ident def_ident(
                expr_direct_call->get_definition( transaction),
                expr_direct_call->get_definition_ident());
            return create_direct_call(
                type.get(), expr_direct_call->get_module(), def_ident,
                expr_direct_call->get_definition_db_name(), clone_arguments.get());
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            return create_temporary( type.get(), expr_temporary->get_index());
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

IExpression_list* Expression_factory::clone(
    const IExpression_list* list,
    DB::Transaction* transaction,
    bool copy_immutable_calls) const
{
    if( !list)
        return nullptr;

    mi::Size n = list->get_size();
    IExpression_list* result = create_expression_list( n);
    for( mi::Size i = 0; i < n; ++i) {

        mi::base::Handle<const IExpression> expr( list->get_expression( i));
        mi::base::Handle<IExpression> clone_expr(
            clone( expr.get(), transaction, copy_immutable_calls));
        const char* name = list->get_name( i);
        result->add_expression_unchecked( name, clone_expr.get());
    }
    return result;
}

namespace {

std::string get_prefix( mi::Size depth)
{
    std::string prefix;
    for( mi::Size i = 0; i < depth; i++)
        prefix += "    ";
    return prefix;
}

} // namespace

const mi::IString* Expression_factory::dump(
    DB::Transaction* transaction,
    const IExpression* expr,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    dump_static( transaction, tf.get(), expr, name, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Expression_factory::dump(
    DB::Transaction* transaction,
    const IExpression_list* list,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    dump_static( transaction, tf.get(), list, name, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Expression_factory::dump(
    DB::Transaction* transaction,
    const IAnnotation* anno,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    dump_static( transaction, tf.get(), anno, name, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Expression_factory::dump(
    DB::Transaction* transaction,
    const IAnnotation_block* block,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    dump_static( transaction, tf.get(), block, name, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Expression_factory::dump(
    DB::Transaction* transaction,
    const IAnnotation_list* list,
    const char* name,
    mi::Size depth) const
{
    std::ostringstream s;
    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    dump_static( transaction, tf.get(), list, name, depth, s);
    return create_istring( s.str().c_str());
}

IExpression* Expression_factory::create_cast(
    DB::Transaction* transaction,
    IExpression* src_expr,
    const IType* target_type,
    const char* cast_db_name,
    bool force_cast,
    bool direct_call,
    mi::Sint32* errors) const
{
    ASSERT( M_SCENE, src_expr);
    ASSERT( M_SCENE, target_type);
    ASSERT( M_SCENE, errors);

    mi::base::Handle<const IType> src_type( src_expr->get_type());
    mi::base::Handle<const IType> stripped_src_type( src_type->skip_all_type_aliases());
    mi::base::Handle<const IType> stripped_target_type( target_type->skip_all_type_aliases());

    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    mi::Sint32 result = tf->is_compatible( stripped_src_type.get(), stripped_target_type.get());
    if( result < 0) {
        *errors = -2;
        return nullptr;
    }
    if( (result == 1) && !force_cast) {
        src_expr->retain();
        return src_expr;
    }

    const char* cast_name = get_cast_operator_db_name();
    DB::Tag cast_def_tag = transaction->name_to_tag( cast_name);
    ASSERT( M_SCENE, cast_def_tag);
    DB::Access<Mdl_function_definition> cast_def( cast_def_tag, transaction);

    mi::base::Handle<IExpression_list> args( create_expression_list( /*initial_capacity*/ 2));
    args->add_expression_unchecked( "cast", src_expr);

    if( direct_call) {
        return create_direct_call(
            stripped_target_type.get(),
            cast_def->get_module( transaction),
            Mdl_tag_ident( cast_def_tag, cast_def->get_ident()),
            cast_name,
            args.get());
    }

    // Create dummy expression for target_type.
    mi::base::Handle<IValue> target_type_value(
        m_value_factory->create( stripped_target_type.get()));
    mi::base::Handle<IExpression> target_type_expr( create_constant( target_type_value.get()));
    args->add_expression_unchecked( "cast_return", target_type_expr.get());

    // Create the function call.
    Mdl_function_call* cast_call = cast_def->create_function_call( transaction, args.get());
    ASSERT( M_SCENE, cast_call);

    // Compute the DB element name.
    std::string call_name;
    if( cast_db_name) {
        DB::Tag tag = transaction->name_to_tag( cast_db_name);
        if( tag)
            call_name = DETAIL::generate_unique_db_name( transaction, cast_db_name);
        else
            call_name = cast_db_name;
    } else
        call_name = DETAIL::generate_unique_db_name( transaction, "mdl::operator_cast");

    // Compute privacy and store level.
    DB::Privacy_level privacy_level = transaction->get_scope()->get_level();
    DB::Tag call_tag = transaction->store_for_reference_counting(
        cast_call, call_name.c_str(), privacy_level);
    ASSERT( M_SCENE, call_tag);

    return create_call( stripped_target_type.get(), call_tag);
}

IExpression* Expression_factory::create_decl_cast(
    DB::Transaction* transaction,
    IExpression* src_expr,
    const IType_struct* target_type,
    const char* cast_db_name,
    bool force_decl_cast,
    bool direct_call,
    mi::Sint32* errors) const
{
    ASSERT( M_SCENE, src_expr);
    ASSERT( M_SCENE, target_type);
    ASSERT( M_SCENE, errors);

    mi::base::Handle<const IType> src_type( src_expr->get_type());
    mi::base::Handle<const IType> stripped_src_type( src_type->skip_all_type_aliases());
    mi::base::Handle<const IType> stripped_target_type( target_type->skip_all_type_aliases());

    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    mi::Sint32 result = tf->from_same_struct_category(
        stripped_src_type.get(), stripped_target_type.get());
    if( result < 0) {
        *errors = -2;
        return nullptr;
    }
    if( (result == 1) && !force_decl_cast) {
        src_expr->retain();
        return src_expr;
    }

    const char* cast_name = get_decl_cast_operator_db_name();
    DB::Tag cast_def_tag = transaction->name_to_tag( cast_name);
    ASSERT( M_SCENE, cast_def_tag);
    DB::Access<Mdl_function_definition> cast_def( cast_def_tag, transaction);

    mi::base::Handle<IExpression_list> args( create_expression_list( /*initial_capacity*/ 2));
    args->add_expression_unchecked( "cast", src_expr);

    if( direct_call) {
        return create_direct_call(
            stripped_target_type.get(),
            cast_def->get_module( transaction),
            Mdl_tag_ident( cast_def_tag, cast_def->get_ident()),
            cast_name,
            args.get());
    }

    // Create dummy expression for target_type.
    mi::base::Handle<IValue> target_type_value(
        m_value_factory->create( stripped_target_type.get()));
    mi::base::Handle<IExpression> target_type_expr( create_constant( target_type_value.get()));
    args->add_expression_unchecked( "cast_return", target_type_expr.get());

    // Create the function call.
    Mdl_function_call* cast_call = cast_def->create_function_call( transaction, args.get());
    ASSERT( M_SCENE, cast_call);

    // Compute the DB element name.
    std::string call_name;
    if( cast_db_name) {
        DB::Tag tag = transaction->name_to_tag( cast_db_name);
        if( tag)
            call_name = DETAIL::generate_unique_db_name( transaction, cast_db_name);
        else
            call_name = cast_db_name;
    } else
        call_name = DETAIL::generate_unique_db_name( transaction, "mdl::operator_decl_cast");

    // Compute privacy and store level.
    DB::Privacy_level privacy_level = 0;
    DB::Privacy_level store_level   = 255;
    if( src_expr->get_kind() == IExpression::EK_CALL) {
        mi::base::Handle<IExpression_call> src_call(
            src_expr->get_interface<IExpression_call>());
        DB::Tag src_call_tag = src_call->get_call();
        privacy_level = transaction->get_tag_privacy_level( src_call_tag);
        store_level = transaction->get_tag_store_level( src_call_tag);
    }

    DB::Tag call_tag = transaction->store_for_reference_counting(
        cast_call, call_name.c_str(), privacy_level, store_level);
    ASSERT( M_SCENE, call_tag);

    return create_call( stripped_target_type.get(), call_tag);
}

void Expression_factory::serialize( SERIAL::Serializer* serializer, const IExpression* expr) const
{
    IExpression::Kind kind = expr->get_kind();
    mi::Uint32 kind_as_uint32 = kind;
    SERIAL::write(serializer,  kind_as_uint32);

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            m_value_factory->serialize( serializer, value.get());
            return;
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            DB::Tag tag = expr_call->get_call();
            SERIAL::write(serializer,  tag);
            return;
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = expr_parameter->get_index();
            SERIAL::write(serializer,  index);
            return;
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag module_tag = expr_direct_call->get_module();
            SERIAL::write(serializer, module_tag);
            DB::Tag tag = expr_direct_call->get_definition(/*transaction*/ nullptr);
            SERIAL::write(serializer,  tag);
            Mdl_ident ident = expr_direct_call->get_definition_ident();
            SERIAL::write(serializer, ident);
            const char* definition_db_name = expr_direct_call->get_definition_db_name();
            SERIAL::write(serializer, definition_db_name);
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            serialize_list( serializer, arguments.get());
            return;
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            mi::Size index = expr_temporary->get_index();
            SERIAL::write(serializer,  index);
            return;
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
    }

    ASSERT( M_SCENE, false);
}

IExpression* Expression_factory::deserialize( SERIAL::Deserializer* deserializer) const
{
    mi::Uint32 kind_as_uint32;
    SERIAL::read(deserializer,  &kind_as_uint32);
    auto kind = static_cast<IExpression::Kind>( kind_as_uint32);

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<IValue> value( m_value_factory->deserialize( deserializer));
            return create_constant( value.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            DB::Tag tag;
            SERIAL::read(deserializer,  &tag);
            return create_call( type.get(), tag);
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            mi::Size index;
            SERIAL::read(deserializer,  &index);
            return create_parameter( type.get(), index);
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            DB::Tag module_tag;
            SERIAL::read(deserializer, &module_tag);
            DB::Tag tag;
            SERIAL::read(deserializer,  &tag);
            Mdl_ident ident;
            SERIAL::read(deserializer, &ident);
            std::string definition_db_name;
            SERIAL::read(deserializer, &definition_db_name);
            mi::base::Handle<IExpression_list> arguments( deserialize_list( deserializer));
            return create_direct_call(
                type.get(),
                module_tag,
                Mdl_tag_ident( tag, ident),
                definition_db_name.c_str(),
                arguments.get());
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            mi::Size index;
            SERIAL::read(deserializer,  &index);
            return create_temporary( type.get(), index);
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    }

    ASSERT( M_SCENE, false);
    return nullptr;
}

void Expression_factory::serialize_list(
    SERIAL::Serializer* serializer, const IExpression_list* list) const
{
    const auto* list_impl = static_cast<const Expression_list*>( list);

    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_expressions.size();
    SERIAL::write(serializer, size);
    for( mi::Size i = 0; i < size; ++i)
        serialize( serializer, list_impl->m_expressions[i].get());
}

IExpression_list* Expression_factory::deserialize_list( SERIAL::Deserializer* deserializer) const
{
    auto* list_impl = new Expression_list( /*initial capacity*/ 0);

    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    SERIAL::read( deserializer,  &size);
    list_impl->m_expressions.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_expressions[i] = deserialize( deserializer);

    return list_impl;
}

void Expression_factory::serialize_annotation(
    SERIAL::Serializer* serializer, const IAnnotation* annotation) const
{
    std::string name = annotation->get_name();
    SERIAL::write( serializer,  name);
    mi::base::Handle<const IExpression_list> arguments( annotation->get_arguments());
    serialize_list( serializer, arguments.get());
}

IAnnotation* Expression_factory::deserialize_annotation(
    SERIAL::Deserializer* deserializer) const
{
    std::string name;
    SERIAL::read( deserializer,  &name);
    mi::base::Handle<IExpression_list> arguments( deserialize_list( deserializer));
    return create_annotation( /*transaction*/ nullptr, name.c_str(), arguments.get());
}

void Expression_factory::serialize_annotation_definition(
    SERIAL::Serializer* serializer, const IAnnotation_definition* anno_def) const
{
    SERIAL::write( serializer, anno_def->get_name());
    SERIAL::write( serializer, anno_def->get_mdl_module_name());
    SERIAL::write( serializer, anno_def->get_mdl_simple_name());

    mi::base::Handle<const IType_list> type_list( anno_def->get_parameter_types());
    size_t n = type_list->get_size();
    serializer->write_size_t( n);
    for(size_t i = 0; i < n; ++i)
        SERIAL::write( serializer, anno_def->get_mdl_parameter_type_name( i));

    SERIAL::write_enum( serializer, anno_def->get_semantic());
    SERIAL::write( serializer, anno_def->is_exported());

    mi::neuraylib::Mdl_version since_version, removed_version;
    anno_def->get_mdl_version( since_version, removed_version);
    write_enum( serializer, since_version);
    write_enum( serializer, removed_version);

    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    tf->serialize_list( serializer, type_list.get());

    mi::base::Handle<const IExpression_list> defaults( anno_def->get_defaults());
    serialize_list( serializer, defaults.get());

    mi::base::Handle<const IAnnotation_block> annotations( anno_def->get_annotations());
    serialize_annotation_block( serializer, annotations.get());
}

IAnnotation_definition* Expression_factory::deserialize_annotation_definition(
    SERIAL::Deserializer* deserializer) const
{
    std::string name;
    SERIAL::read( deserializer, &name);

    std::string mdl_module_name;
    SERIAL::read( deserializer, &mdl_module_name);

    std::string mdl_simple_name;
    SERIAL::read( deserializer, &mdl_simple_name);

    size_t n;
    deserializer->read_size_t( &n);
    std::vector<std::string> core_parameter_type_names;
    for( size_t i = 0; i < n; ++i) {
        std::string s;
        SERIAL::read( deserializer, &s);
        core_parameter_type_names.push_back( s);
    }

    mi::neuraylib::IAnnotation_definition::Semantics semantic;
    SERIAL::read_enum( deserializer, &semantic);

    bool is_exported = false;
    SERIAL::read( deserializer, &is_exported);

    mi::neuraylib::Mdl_version since_version, removed_version;
    read_enum( deserializer, &since_version);
    read_enum( deserializer, &removed_version);

    mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
    mi::base::Handle<IType_list> parameter_types( tf->deserialize_list( deserializer));
    mi::base::Handle<IExpression_list> defaults( deserialize_list( deserializer));
    mi::base::Handle<IAnnotation_block> annotations( deserialize_annotation_block( deserializer));

    return create_annotation_definition(
        name.c_str(),
        mdl_module_name.c_str(),
        mdl_simple_name.c_str(),
        core_parameter_type_names,
        semantic,
        is_exported,
        since_version,
        removed_version,
        parameter_types.get(),
        defaults.get(),
        annotations.get());
}

void Expression_factory::serialize_annotation_block(
    SERIAL::Serializer* serializer, const IAnnotation_block* block) const
{
    mi::Size size = block ? block->get_size() : 0;
    SERIAL::write( serializer, size);
    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<const IAnnotation> anno( block->get_annotation( i));
        serialize_annotation( serializer, anno.get());
    }
}

IAnnotation_block* Expression_factory::deserialize_annotation_block(
    SERIAL::Deserializer* deserializer) const
{
    mi::Size size;
    SERIAL::read( deserializer, &size);
    if( size == 0)
        return nullptr;

    IAnnotation_block* block = new Annotation_block( size);
    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<IAnnotation> anno( deserialize_annotation( deserializer));
        block->add_annotation( anno.get());
    }
    return block;
}

void Expression_factory::serialize_annotation_list(
    SERIAL::Serializer* serializer, const IAnnotation_list* list) const
{
    const auto* list_impl = static_cast<const Annotation_list*>( list);

    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_annotation_blocks.size();
    SERIAL::write( serializer, size);
    for( mi::Size i = 0; i < size; ++i)
        serialize_annotation_block( serializer, list_impl->m_annotation_blocks[i].get());
}

IAnnotation_list* Expression_factory::deserialize_annotation_list(
    SERIAL::Deserializer* deserializer) const
{
    auto* list_impl = new Annotation_list( /*initial_capacity*/ 0);

    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    SERIAL::read( deserializer, &size);
    list_impl->m_annotation_blocks.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_annotation_blocks[i] = deserialize_annotation_block( deserializer);

    return list_impl;
}

void Expression_factory::serialize_annotation_definition_list(
    SERIAL::Serializer* serializer,
    const IAnnotation_definition_list* anno_def_list) const
{
    const auto* anno_def_list_impl = static_cast<const Annotation_definition_list*>( anno_def_list);

    mi::Size size = anno_def_list_impl->m_anno_definitions.size();
    SERIAL::write( serializer, size);
    for( mi::Size i = 0; i < size; ++i)
        serialize_annotation_definition(
            serializer, anno_def_list_impl->m_anno_definitions[i].get());
}

IAnnotation_definition_list* Expression_factory::deserialize_annotation_definition_list(
    SERIAL::Deserializer* deserializer) const
{
    auto* list_impl = new Annotation_definition_list( /*initial_capacity*/ 0);

    mi::Size size;
    SERIAL::read( deserializer, &size);
    list_impl->m_anno_definitions.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_anno_definitions[i] = deserialize_annotation_definition( deserializer);

    return list_impl;
}

void Expression_factory::dump_static(
    DB::Transaction* transaction,
    IType_factory* tf,
    const IExpression* expr,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !expr)
        return;

    IExpression::Kind kind = expr->get_kind();

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> expr_constant(
                expr->get_interface<IExpression_constant>());
            s << "constant ";
            mi::base::Handle<const IValue> value( expr_constant->get_value());
            Value_factory::dump_static( transaction, value.get(), name, depth, s);
            return;
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> expr_call(
                expr->get_interface<IExpression_call>());
            s << "call ";
#if 0 // sometimes useful, but generates too much output
            mi::base::Handle<const IType> type( expr_call->get_type());
            mi::base::Handle<const mi::IString> type_dumped( tf->dump( type.get(), depth));
            s << type_dumped->get_c_str() << ' ';
#endif
            if( name)
                s << name << " = ";
            DB::Tag tag = expr_call->get_call();
            if( !tag) {
                s << "(unset)";
                return;
            }
            if( transaction)
                s << '\"' << transaction->tag_to_name( tag) << '\"';
            else
                s << "tag " << tag.get_uint();
            return;
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            s << "parameter ";
            mi::base::Handle<const IType> type( expr_parameter->get_type());
            mi::base::Handle<const mi::IString> type_dumped( tf->dump( type.get(), depth));
            s << type_dumped->get_c_str() << ' ';
            if( name)
                s << name << " = ";
            mi::Size index = expr_parameter->get_index();
            s << "index " << index;
            return;
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            s << "direct call ";
#if 0 // sometimes useful, but generates too much output
            mi::base::Handle<const IType> type( expr_direct_call->get_type());
            mi::base::Handle<const mi::IString> type_dumped( tf->dump( type.get(), depth));
            s << type_dumped->get_c_str() << ' ';
#endif
            if( name)
                s << name << " = ";
            DB::Tag tag = expr_direct_call->get_definition(/*transaction*/ nullptr);
            if( transaction)
                s << '\"' << transaction->tag_to_name( tag) << "\" (";
            else
                s << "tag " << tag.get_uint() << " (";
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            mi::Size n = arguments->get_size();
            s << (n > 0 ? "\n" : "");
            const std::string& prefix = get_prefix( depth);
            for( mi::Size i = 0; i < n; ++i) {
                const char* arg_name = arguments->get_name( i);
                mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
                s << prefix << "    ";
                dump_static( transaction, tf, argument.get(), arg_name, depth+1, s);
                s << (i < n-1 ? ",\n" : "\n");
            }
            s << (n > 0 ? prefix : "") << ')';
            return;
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            s << "temporary ";
#if 0 // sometimes useful, but generates too much output
            mi::base::Handle<const IType> type( expr_temporary->get_type());
            mi::base::Handle<const mi::IString> type_dumped( tf->dump( type.get(), depth));
            s << type_dumped->get_c_str() << ' ';
#endif
            if( name)
                s << name << " = ";
            mi::Size index = expr_temporary->get_index();
            s << "index " << index;
            return;
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
}

void Expression_factory::dump_static(
    DB::Transaction* transaction,
    IType_factory* tf,
    const IExpression_list* list,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !list)
        return;

    s << "expression_list ";
    if( name)
        s << name << " = ";

    mi::Size n = list->get_size();
    s << (n > 0 ? "[\n" : "[ ");

    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IExpression> expr( list->get_expression( i));
        s << prefix << "    ";
        std::ostringstream elem_name;
        elem_name << i << ": " << list->get_name( i);
        dump_static( transaction, tf, expr.get(), elem_name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << ']';
}

void Expression_factory::dump_static(
    DB::Transaction* transaction,
    IType_factory* tf,
    const IAnnotation* anno,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !anno)
        return;

    s << "annotation ";
    if( name)
        s << name << " = ";
    s << '\"' << anno->get_name() << "\" (";

    mi::base::Handle<const IExpression_list> arguments( anno->get_arguments());
    mi::Size n = arguments->get_size();
    s << (n > 0 ? "\n" : "");
    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        const char* arg_name = arguments->get_name( i);
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        s << prefix << "    ";
        dump_static( transaction, tf, argument.get(), arg_name, depth+1, s);
        s << (i < n-1 ? ",\n" : "\n");
    }
    s << (n > 0 ? prefix : "") << ')';
}

void Expression_factory::dump_static(
    DB::Transaction* transaction,
    IType_factory* tf,
    const IAnnotation_block* block,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !block)
        return;

    s << "annotation_block ";
    if( name)
        s << name << " = ";

    mi::Size n = block->get_size();
    s << (n > 0 ? "[\n" : "[ ");

    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IAnnotation> anno( block->get_annotation( i));
        s << prefix << "    ";
        std::ostringstream anno_name;
        anno_name << i;
        dump_static( transaction, tf, anno.get(), anno_name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << "]";
}

void Expression_factory::dump_static(
    DB::Transaction* transaction,
    IType_factory* tf,
    const IAnnotation_list* list,
    const char* name,
    mi::Size depth,
    std::ostringstream& s)
{
    if( !list)
        return;

    s << "annotation_list ";
    if( name)
        s << name << " = ";

    mi::Size n = list->get_size();
    s << (n > 0 ? "[\n" : "[ ");

    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IAnnotation_block> block( list->get_annotation_block( i));
        s << prefix << "    ";
        std::ostringstream anno_name;
        anno_name << i << ": " << list->get_name( i);
        dump_static( transaction, tf, block.get(), anno_name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << "]";
}

mi::Sint32 Expression_factory::compare_static(
    const IExpression* lhs,
    const IExpression* rhs,
    mi::Uint32 flags,
    mi::Float64 epsilon,
    DB::Transaction* transaction)
{
    ASSERT( M_SCENE,
        transaction || ((flags & mi::neuraylib::IExpression_factory::DEEP_CALL_COMPARISONS) == 0));

    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::base::Handle<const IType> lhs_type( lhs->get_type()); //-V522 PVS
    mi::base::Handle<const IType> rhs_type( rhs->get_type()); //-V522 PVS
    if( (flags & mi::neuraylib::IExpression_factory::SKIP_TYPE_ALIASES) != 0) {
        lhs_type = lhs_type->skip_all_type_aliases();
        rhs_type = rhs_type->skip_all_type_aliases();
    }
    mi::Sint32 result = Type_factory::compare_static( lhs_type.get(), rhs_type.get());
    if( result != 0)
        return result;

    IExpression::Kind lhs_kind = lhs->get_kind();
    IExpression::Kind rhs_kind = rhs->get_kind();
    if( lhs_kind < rhs_kind) return -1;
    if( lhs_kind > rhs_kind) return +1;

    switch( lhs_kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<const IExpression_constant> lhs_constant(
                lhs->get_interface<IExpression_constant>());
            mi::base::Handle<const IExpression_constant> rhs_constant(
                rhs->get_interface<IExpression_constant>());
            mi::base::Handle<const IValue> lhs_value( lhs_constant->get_value());
            mi::base::Handle<const IValue> rhs_value( rhs_constant->get_value());
            return Value_factory::compare_static(
                lhs_value.get(), rhs_value.get(), epsilon);
        }

        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> lhs_call(
                lhs->get_interface<IExpression_call>());
            mi::base::Handle<const IExpression_call> rhs_call(
                rhs->get_interface<IExpression_call>());
            DB::Tag lhs_tag = lhs_call->get_call();
            DB::Tag rhs_tag = rhs_call->get_call();

            if( (flags & mi::neuraylib::IExpression_factory::DEEP_CALL_COMPARISONS) != 0) {

                // Get referenced function calls
                SERIAL::Class_id lhs_class_id = transaction->get_class_id( lhs_tag);
                SERIAL::Class_id rhs_class_id = transaction->get_class_id( rhs_tag);
                if( lhs_class_id < rhs_class_id) return -1;
                if( lhs_class_id > rhs_class_id) return +1;
                // Not much we can do if both class IDs are equal and wrong.
                if( lhs_class_id != ID_MDL_FUNCTION_CALL) return 0;
                DB::Access<Mdl_function_call> lhs_fc( lhs_tag, transaction);
                DB::Access<Mdl_function_call> rhs_fc( rhs_tag, transaction);

                // Compare underlying function definition
                DB::Tag lhs_fd_tag = lhs_fc->get_function_definition( transaction);
                DB::Tag rhs_fd_tag = rhs_fc->get_function_definition( transaction);
                if( lhs_fd_tag < rhs_fd_tag) return -1;
                if( lhs_fd_tag > rhs_fd_tag) return +1;

                // Compare arguments
                mi::base::Handle<const IExpression_list> lhs_arguments( lhs_fc->get_arguments());
                mi::base::Handle<const IExpression_list> rhs_arguments( rhs_fc->get_arguments());
                return compare_static(
                    lhs_arguments.get(), rhs_arguments.get(), flags, epsilon, transaction);

            } else {

                // Simply compare the tags
                if( lhs_tag < rhs_tag) return -1;
                if( lhs_tag > rhs_tag) return +1;
                return 0;

            }
        }

        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IExpression_parameter> lhs_parameter(
                lhs->get_interface<IExpression_parameter>());
            mi::base::Handle<const IExpression_parameter> rhs_parameter(
                rhs->get_interface<IExpression_parameter>());
            mi::Size lhs_index = lhs_parameter->get_index();
            mi::Size rhs_index = rhs_parameter->get_index();
            if( lhs_index < rhs_index) return -1;
            if( lhs_index > rhs_index) return +1;
            return 0;
        }

        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> lhs_direct_call(
                lhs->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IExpression_direct_call> rhs_direct_call(
                rhs->get_interface<IExpression_direct_call>());
            DB::Tag lhs_tag = lhs_direct_call->get_definition(/*transaction*/ nullptr);
            DB::Tag rhs_tag = rhs_direct_call->get_definition(/*transaction*/ nullptr);
            if( lhs_tag < rhs_tag) return -1;
            if( lhs_tag > rhs_tag) return +1;
            DB::Tag lhs_mtag = lhs_direct_call->get_module();
            DB::Tag rhs_mtag = rhs_direct_call->get_module();
            if (lhs_mtag < rhs_mtag) return -1;
            if (lhs_mtag > rhs_mtag) return +1;
            mi::base::Handle<const IExpression_list> lhs_arguments(
                lhs_direct_call->get_arguments());
            mi::base::Handle<const IExpression_list> rhs_arguments(
                rhs_direct_call->get_arguments());
            return compare_static(
                lhs_arguments.get(), rhs_arguments.get(), flags, epsilon, transaction);
        }

        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> lhs_temporary(
                lhs->get_interface<IExpression_temporary>());
            mi::base::Handle<const IExpression_temporary> rhs_temporary(
                rhs->get_interface<IExpression_temporary>());
            mi::Size lhs_index = lhs_temporary->get_index();
            mi::Size rhs_index = rhs_temporary->get_index();
            if( lhs_index < rhs_index) return -1;
            if( lhs_index > rhs_index) return +1;
            return 0;
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

mi::Sint32 Expression_factory::compare_static(
    const IExpression_list* lhs,
    const IExpression_list* rhs,
    mi::Uint32 flags,
    mi::Float64 epsilon,
    DB::Transaction* transaction)
{
    ASSERT( M_SCENE,
        transaction || ((flags & mi::neuraylib::IExpression_factory::DEEP_CALL_COMPARISONS) == 0));

    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::Size lhs_n = lhs->get_size(); //-V522 PVS
    mi::Size rhs_n = rhs->get_size(); //-V522 PVS
    if( lhs_n < rhs_n) return -1;
    if( lhs_n > rhs_n) return +1;

    for( mi::Size i = 0; i < lhs_n; ++i) {
        const char* lhs_name = lhs->get_name( i);
        const char* rhs_name = rhs->get_name( i);
        mi::Sint32 result = strcmp( lhs_name, rhs_name);
        if( result < 0) return -1;
        if( result > 0) return +1;
        mi::base::Handle<const IExpression> lhs_expression( lhs->get_expression( i));
        mi::base::Handle<const IExpression> rhs_expression( rhs->get_expression( i));
        result = compare_static(
            lhs_expression.get(), rhs_expression.get(), flags, epsilon, transaction);
        if( result != 0)
            return result;
    }

    return 0;
}

Annotation_definition_list::Annotation_definition_list( mi::Size initial_capacity)
{
    m_anno_definitions.reserve( initial_capacity);
}

const IAnnotation_definition* Annotation_definition_list::get_definition( mi::Size index) const
{
    if( index >= m_anno_definitions.size())
        return nullptr;

    m_anno_definitions[index]->retain();
    return m_anno_definitions[index].get();
}

const IAnnotation_definition* Annotation_definition_list::get_definition( const char* name) const
{
    if( !name)
        return nullptr;

    // For typical list sizes a linear search is much faster than maintaining a map from names to
    // indices.
    for( const auto& anno_def: m_anno_definitions)
        if( strcmp( anno_def->get_name(), name) == 0) {
            anno_def->retain();
            return anno_def.get();
        }

    return nullptr;
}

mi::Sint32 Annotation_definition_list::add_definition( const IAnnotation_definition* anno_def)
{
    if( !anno_def)
        return -1;

    // A linear search is much faster than maintaining a map from names to indices.
    const char* name = anno_def->get_name();
    for( const auto& anno_def: m_anno_definitions)
        if( strcmp( anno_def->get_name(), name) == 0)
            return -2;

    m_anno_definitions.push_back( mi::base::make_handle_dup( anno_def));
    return 0;
}

void Annotation_definition_list::add_definition_unchecked(
    const IAnnotation_definition* anno_def)
{
    m_anno_definitions.push_back( mi::base::make_handle_dup( anno_def));
}

mi::Size Annotation_definition_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_anno_definitions);
}


class Factories
{
public:
    Factories()
      : m_tf( new Type_factory()),
        m_vf( new Value_factory( m_tf.get())),
        m_ef( new Expression_factory( m_vf.get())) { }

    mi::base::Handle<Type_factory> m_tf;
    mi::base::Handle<Value_factory> m_vf;
    mi::base::Handle<Expression_factory> m_ef;
};

Factories g_factories;

IType_factory* get_type_factory()
{
   g_factories.m_tf->retain();
   return g_factories.m_tf.get();
}

IValue_factory* get_value_factory()
{
   g_factories.m_vf->retain();
   return g_factories.m_vf.get();
}

IExpression_factory* get_expression_factory()
{
   g_factories.m_ef->retain();
   return g_factories.m_ef.get();
}

} // namespace MDL

} // namespace MI
