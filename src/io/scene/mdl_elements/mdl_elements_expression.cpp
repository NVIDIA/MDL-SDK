/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "i_mdl_elements_material_instance.h"
#include "i_mdl_elements_function_call.h"
#include "mdl_elements_expression.h"
#include "mdl_elements_type.h"
#include "mdl_elements_value.h"
#include "mdl_elements_utilities.h"

#include <mi/neuraylib/istring.h>
#include <cstring>
#include <sstream>
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

mi::Size Expression_list::get_size() const
{
    return m_expressions.size();
}

mi::Size Expression_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);
    Name_index_map::const_iterator it = m_name_index.find( name);
    if( it == m_name_index.end())
        return static_cast<mi::Size>( -1);
    return it->second;
}

const char* Expression_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return 0;
    return m_index_name[index].c_str();
}

const IExpression* Expression_list::get_expression( mi::Size index) const
{
    if( index >= m_expressions.size())
        return 0;
    m_expressions[index]->retain();
    return m_expressions[index].get();
}

const IExpression* Expression_list::get_expression( const char* name) const
{
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return 0;
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
    m_name_index[name] = m_expressions.size() - 1;
    m_index_name.push_back( name);
    return 0;
}

mi::Size Expression_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_name_index)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_expressions);
}

const IExpression_list* Annotation::get_arguments() const
{
    m_arguments->retain();
    return m_arguments.get();
}

mi::Size Annotation::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_name)
        + dynamic_memory_consumption( m_arguments);
}

mi::Size Annotation_block::get_size() const
{
    return m_annotations.size();
}

const IAnnotation* Annotation_block::get_annotation( mi::Size index) const
{
    if( index >= m_annotations.size())
        return 0;
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

mi::Size Annotation_list::get_size() const
{
    return m_annotation_blocks.size();
}

mi::Size Annotation_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);
    Name_index_map::const_iterator it = m_name_index.find( name);
    if( it == m_name_index.end())
        return static_cast<mi::Size>( -1);
    return it->second;
}

const char* Annotation_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return 0;
    return m_index_name[index].c_str();
}

const IAnnotation_block* Annotation_list::get_annotation_block( mi::Size index) const
{
    if( index >= m_annotation_blocks.size())
        return 0;
    m_annotation_blocks[index]->retain();
    return m_annotation_blocks[index].get();
}

const IAnnotation_block* Annotation_list::get_annotation_block( const char* name) const
{
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return 0;
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
    m_name_index[name] = m_annotation_blocks.size() - 1;
    m_index_name.push_back( name);
    return 0;
}

mi::Size Annotation_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_name_index)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_annotation_blocks);
}

Expression_factory::Expression_factory( IValue_factory* value_factory)
  : m_value_factory( value_factory, mi::base::DUP_INTERFACE)
{
}

Expression_factory::~Expression_factory()
{
}

IValue_factory* Expression_factory::get_value_factory() const
{
    m_value_factory->retain();
    return m_value_factory.get();
}

IExpression_constant* Expression_factory::create_constant( IValue* value) const
{
    return value ? new Expression_constant( value) : 0;
}

IExpression_call* Expression_factory::create_call( const IType* type, DB::Tag tag) const
{
    return type && tag ? new Expression_call( type, tag) : 0;
}

IExpression_parameter* Expression_factory::create_parameter(
    const IType* type, mi::Size index) const
{
    return type ? new Expression_parameter( type, index) : 0;
}

IExpression_direct_call* Expression_factory::create_direct_call(
    const IType* type, DB::Tag tag, IExpression_list* arguments) const
{
    return type && tag && arguments ? new Expression_direct_call( type, tag, arguments) : 0;
}

IExpression_temporary* Expression_factory::create_temporary(
    const IType* type, mi::Size index) const
{
    return type ? new Expression_temporary( type, index) : 0;
}

IAnnotation* Expression_factory::create_annotation(
    const char* name, const IExpression_list* arguments) const
{
    if( !name || !arguments)
        return 0;

    mi::Size n = arguments->get_size();
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        if( argument->get_kind() != IExpression::EK_CONSTANT)
            return 0;
    }

    return new Annotation( name, arguments);
}

IExpression_list* Expression_factory::create_expression_list() const
{
    return new Expression_list;
}

IAnnotation_block* Expression_factory::create_annotation_block() const
{
    return new Annotation_block;
}

IAnnotation_list* Expression_factory::create_annotation_list() const
{
    return new Annotation_list;
}

IExpression* Expression_factory::clone(
    const IExpression* expr, 
    DB::Transaction* transaction,
    bool copy_immutable_calls) const
{
    if( !expr)
        return 0;

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
            if (copy_immutable_calls)
            {
                ASSERT(M_SCENE, transaction);
                DB::Tag call_tag = expr_call->get_call();
                SERIAL::Class_id class_id = transaction->get_class_id(call_tag);
                std::vector<mi::base::Handle<const IExpression> > dummy_context;
                if (class_id == Mdl_function_call::id) {
                    
                    DB::Access<Mdl_function_call> f_call(call_tag, transaction);
                    if (f_call->is_immutable())
                        return deep_copy(this, transaction, expr_call.get(), dummy_context);
                }
                else if (class_id == Mdl_material_instance::id) {
                    DB::Access<Mdl_material_instance> mat_inst(call_tag, transaction);
                    if (mat_inst->is_immutable())
                        return deep_copy(this, transaction, expr_call.get(), dummy_context);
                }
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
            return create_direct_call(
                type.get(), expr_direct_call->get_definition(), clone_arguments.get());
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            return create_temporary( type.get(), expr_temporary->get_index());
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

IExpression_list* Expression_factory::clone(
    const IExpression_list* list,
    DB::Transaction* transaction,
    bool copy_immutable_calls) const
{
    if( !list)
        return 0;

    IExpression_list* result = create_expression_list();
    mi::Size n = list->get_size();
    for( mi::Size i = 0; i < n; ++i) {

        mi::base::Handle<const IExpression> expr( list->get_expression( i));
        mi::base::Handle<IExpression> clone_expr(
            clone( expr.get(), transaction, copy_immutable_calls));
        const char* name = list->get_name( i);
        result->add_expression( name, clone_expr.get());
    }
    return result;
}

namespace {

class String : public mi::base::Interface_implement<mi::IString>
{
public:
    String( const char* str = 0) : m_string( str ? str : "") { }
    const char* get_type_name() const { return "String"; }
    const char* get_c_str() const { return m_string.c_str(); }
    void set_c_str( const char* str) { m_string = str ? str : ""; }
private:
    std::string m_string;
};

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
    return new String( s.str().c_str());
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
    return new String( s.str().c_str());
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
    return new String( s.str().c_str());
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
    return new String( s.str().c_str());
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
    return new String( s.str().c_str());
}

void Expression_factory::serialize( SERIAL::Serializer* serializer, const IExpression* expr) const
{
    IExpression::Kind kind = expr->get_kind();
    mi::Uint32 kind_as_uint32 = kind;
    serializer->write( kind_as_uint32);

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
            serializer->write( tag);
            return;
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_parameter> expr_parameter(
                expr->get_interface<IExpression_parameter>());
            mi::Size index = expr_parameter->get_index();
            serializer->write( index);
            return;
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IType> type( expr->get_type());
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            tf->serialize( serializer, type.get());
            mi::base::Handle<const IExpression_direct_call> expr_direct_call(
                expr->get_interface<IExpression_direct_call>());
            DB::Tag tag = expr_direct_call->get_definition();
            serializer->write( tag);
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
            serializer->write( index);
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
    deserializer->read( &kind_as_uint32);
    IExpression::Kind kind = static_cast<IExpression::Kind>( kind_as_uint32);

    switch( kind) {

        case IExpression::EK_CONSTANT: {
            mi::base::Handle<IValue> value( m_value_factory->deserialize( deserializer));
            return create_constant( value.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            DB::Tag tag;
            deserializer->read( &tag);
            return create_call( type.get(), tag);
        }
        case IExpression::EK_PARAMETER: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            mi::Size index;
            deserializer->read( &index);
            return create_parameter( type.get(), index);
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            DB::Tag tag;
            deserializer->read( &tag);
            mi::base::Handle<IExpression_list> arguments( deserialize_list( deserializer));
            return create_direct_call( type.get(), tag, arguments.get());
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<IType_factory> tf( m_value_factory->get_type_factory());
            mi::base::Handle<const IType> type( tf->deserialize( deserializer));
            mi::Size index;
            deserializer->read( &index);
            return create_temporary( type.get(), index);
        }
        case IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

void Expression_factory::serialize_list(
    SERIAL::Serializer* serializer, const IExpression_list* list) const
{
    const Expression_list* list_impl = static_cast<const Expression_list*>( list);

    write( serializer, list_impl->m_name_index);
    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_expressions.size();
    serializer->write( size);
    for( mi::Size i = 0; i < size; ++i)
        serialize( serializer, list_impl->m_expressions[i].get());
}

IExpression_list* Expression_factory::deserialize_list( SERIAL::Deserializer* deserializer) const
{
    Expression_list* list_impl = new Expression_list;

    read( deserializer, &list_impl->m_name_index);
    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    deserializer->read( &size);
    list_impl->m_expressions.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_expressions[i] = deserialize( deserializer);

    return list_impl;
}

void Expression_factory::serialize_annotation(
    SERIAL::Serializer* serializer, const IAnnotation* annotation) const
{
    std::string name = annotation->get_name();
    serializer->write( name);
    mi::base::Handle<const IExpression_list> arguments( annotation->get_arguments());
    serialize_list( serializer, arguments.get());
}

IAnnotation* Expression_factory::deserialize_annotation(
    SERIAL::Deserializer* deserializer) const
{
    std::string name;
    deserializer->read( &name);
    mi::base::Handle<IExpression_list> arguments( deserialize_list( deserializer));
    return create_annotation( name.c_str(), arguments.get());
}

void Expression_factory::serialize_annotation_block(
    SERIAL::Serializer* serializer, const IAnnotation_block* block) const
{
    mi::Size size = block ? block->get_size() : 0;
    serializer->write( size);
    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<const IAnnotation> anno( block->get_annotation( i));
        serialize_annotation( serializer, anno.get());
    }
}

IAnnotation_block* Expression_factory::deserialize_annotation_block(
    SERIAL::Deserializer* deserializer) const
{
    mi::Size size;
    deserializer->read( &size);
    if( size == 0)
        return 0;

    IAnnotation_block* block = new Annotation_block;
    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<IAnnotation> anno( deserialize_annotation( deserializer));
        block->add_annotation( anno.get());
    }
    return block;
}

void Expression_factory::serialize_annotation_list(
    SERIAL::Serializer* serializer, const IAnnotation_list* list) const
{
    const Annotation_list* list_impl = static_cast<const Annotation_list*>( list);

    write( serializer, list_impl->m_name_index);
    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_annotation_blocks.size();
    serializer->write( size);
    for( mi::Size i = 0; i < size; ++i)
        serialize_annotation_block( serializer, list_impl->m_annotation_blocks[i].get());
}

IAnnotation_list* Expression_factory::deserialize_annotation_list(
    SERIAL::Deserializer* deserializer) const
{
    Annotation_list* list_impl = new Annotation_list;

    read( deserializer, &list_impl->m_name_index);
    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    deserializer->read( &size);
    list_impl->m_annotation_blocks.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_annotation_blocks[i] = deserialize_annotation_block( deserializer);

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
            s << type_dumped->get_c_str() << " ";
#endif
            if( name)
                s << name << " = ";
            DB::Tag tag = expr_call->get_call();
            if( !tag) {
                s << "(unset)";
                return;
            }
            if( transaction)
                s << "\"" << transaction->tag_to_name( tag) << "\"";
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
            s << type_dumped->get_c_str() << " ";
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
            s << type_dumped->get_c_str() << " ";
#endif
            if( name)
                s << name << " = ";
            DB::Tag tag = expr_direct_call->get_definition();
            if( transaction)
                s << "\"" << transaction->tag_to_name( tag) << "\" (";
            else
                s << "tag " << tag.get_uint() << " (";
            mi::base::Handle<const IExpression_list> arguments( expr_direct_call->get_arguments());
            mi::Size n = arguments->get_size();
            s << (n > 0 ? "\n" : "");
            const std::string& prefix = get_prefix( depth);
            for( mi::Size i = 0; i < n; ++i) {
                const char* name = arguments->get_name( i);
                mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
                s << prefix << "    ";
                dump_static( transaction, tf, argument.get(), name, depth+1, s);
                s << (i < n-1 ? ",\n" : "\n");
            }
            s << (n > 0 ? prefix : "") << ")";
            return;
        }
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> expr_temporary(
                expr->get_interface<IExpression_temporary>());
            s << "temporary ";
#if 0 // sometimes useful, but generates too much output
            mi::base::Handle<const IType> type( expr_temporary->get_type());
            mi::base::Handle<const mi::IString> type_dumped( tf->dump( type.get(), depth));
            s << type_dumped->get_c_str() << " ";
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
        std::ostringstream name;
        name << i << ": " << list->get_name( i);
        dump_static( transaction, tf, expr.get(), name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << "]";
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
    s << "\"" << anno->get_name() << "\" (";

    mi::base::Handle<const IExpression_list> arguments( anno->get_arguments());
    mi::Size n = arguments->get_size();
    s << (n > 0 ? "\n" : "");
    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        const char* name = arguments->get_name( i);
        mi::base::Handle<const IExpression> argument( arguments->get_expression( i));
        s << prefix << "    ";
        dump_static( transaction, tf, argument.get(), name, depth+1, s);
        s << (i < n-1 ? ",\n" : "\n");
    }
    s << (n > 0 ? prefix : "") << ")";
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
        std::ostringstream name;
        name << i;
        dump_static( transaction, tf, anno.get(), name.str().c_str(), depth+1, s);
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
        std::ostringstream name;
        name << i << ": " << list->get_name( i);
        dump_static( transaction, tf, block.get(), name.str().c_str(), depth+1, s);
        s << ";\n";
    }

    s << (n > 0 ? prefix : "") << "]";
}

mi::Sint32 Expression_factory::compare_static( const IExpression* lhs, const IExpression* rhs)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::base::Handle<const IType> lhs_type( lhs->get_type());
    mi::base::Handle<const IType> rhs_type( rhs->get_type());
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
            return Value_factory::compare_static( lhs_value.get(), rhs_value.get());
        }
        case IExpression::EK_CALL: {
            mi::base::Handle<const IExpression_call> lhs_call(
                lhs->get_interface<IExpression_call>());
            mi::base::Handle<const IExpression_call> rhs_call(
                rhs->get_interface<IExpression_call>());
            DB::Tag lhs_tag = lhs_call->get_call();
            DB::Tag rhs_tag = rhs_call->get_call();
            if( lhs_tag < rhs_tag) return -1;
            if( lhs_tag > rhs_tag) return +1;
            return 0;
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
            DB::Tag lhs_tag = lhs_direct_call->get_definition();
            DB::Tag rhs_tag = rhs_direct_call->get_definition();
            if( lhs_tag < rhs_tag) return -1;
            if( lhs_tag > rhs_tag) return +1;
            mi::base::Handle<const IExpression_list> lhs_arguments(
                lhs_direct_call->get_arguments());
            mi::base::Handle<const IExpression_list> rhs_arguments(
                rhs_direct_call->get_arguments());
            return compare_static( lhs_arguments.get(), rhs_arguments.get());
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
    const IExpression_list* lhs, const IExpression_list* rhs)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::Size lhs_n = lhs->get_size();
    mi::Size rhs_n = rhs->get_size();
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
        result = compare_static( lhs_expression.get(), rhs_expression.get());
        if( result != 0)
            return result;
    }

    return 0;
}

class Factories
{
public:
    Factories() : m_tf(), m_vf( &m_tf), m_ef( &m_vf) { }
    Type_factory m_tf;
    Value_factory m_vf;
    Expression_factory m_ef;
};

Factories g_factories;

IType_factory* get_type_factory()
{
   g_factories.m_tf.retain();
   return &g_factories.m_tf;
}

IValue_factory* get_value_factory()
{
   g_factories.m_vf.retain();
   return &g_factories.m_vf;
}

IExpression_factory* get_expression_factory()
{
   g_factories.m_ef.retain();
   return &g_factories.m_ef;
}

} // namespace MDL

} // namespace MI
