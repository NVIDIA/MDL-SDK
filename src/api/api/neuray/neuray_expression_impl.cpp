/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IExpression implementation.
 **/

#include "pch.h"

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <io/scene/mdl_elements/i_mdl_elements_expression.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include "neuray_expression_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_value_impl.h"
#include "neuray_type_impl.h"

namespace MI {

namespace NEURAY {

MDL::IExpression* get_internal_expression( mi::neuraylib::IExpression* expr)
{
    if( !expr)
        return nullptr;
    mi::base::Handle<IExpression_wrapper> expr_wrapper(
        expr->get_interface<IExpression_wrapper>());
    if( !expr_wrapper)
        return nullptr;
    return expr_wrapper->get_internal_expression();
}

const MDL::IExpression* get_internal_expression( const mi::neuraylib::IExpression* expr)
{
    if( !expr)
        return nullptr;
    mi::base::Handle<const IExpression_wrapper> expr_wrapper(
         expr->get_interface<IExpression_wrapper>());
    if( !expr_wrapper)
        return nullptr;
    return expr_wrapper->get_internal_expression();
}

MDL::IExpression_list* get_internal_expression_list( mi::neuraylib::IExpression_list* expr_list)
{
    if( !expr_list)
        return nullptr;
    mi::base::Handle<IExpression_list_wrapper> expr_list_wrapper(
        expr_list->get_interface<IExpression_list_wrapper>());
    if( !expr_list_wrapper)
        return nullptr;
    return expr_list_wrapper->get_internal_expression_list();
}

const MDL::IExpression_list* get_internal_expression_list(
    const mi::neuraylib::IExpression_list* expr_list)
{
    if( !expr_list)
        return nullptr;
    mi::base::Handle<const IExpression_list_wrapper> expr_list_wrapper(
        expr_list->get_interface<IExpression_list_wrapper>());
    if( !expr_list_wrapper)
        return nullptr;
    return expr_list_wrapper->get_internal_expression_list();
}

MDL::IAnnotation* get_internal_annotation( mi::neuraylib::IAnnotation* anno)
{
    if( !anno)
        return nullptr;
    Annotation* anno_int = static_cast<Annotation*>( anno);
    return anno_int->get_internal_annotation();
}

const MDL::IAnnotation* get_internal_annotation( const mi::neuraylib::IAnnotation* anno)
{
    if( !anno)
        return nullptr;
    const Annotation* anno_int = static_cast<const Annotation*>( anno);
    return anno_int->get_internal_annotation();
}

MDL::IAnnotation_block* get_internal_annotation_block( mi::neuraylib::IAnnotation_block* block)
{
    if( !block)
        return nullptr;
    Annotation_block* block_int = static_cast<Annotation_block*>( block);
    return block_int->get_internal_annotation_block();
}

const MDL::IAnnotation_block* get_internal_annotation_block(
    const mi::neuraylib::IAnnotation_block* block)
{
    if( !block)
        return nullptr;
    const Annotation_block* block_int = static_cast<const Annotation_block*>( block);
    return block_int->get_internal_annotation_block();
}

MDL::IAnnotation_list* get_internal_annotation_list( mi::neuraylib::IAnnotation_list* list)
{
    if( !list)
        return nullptr;
    Annotation_list* list_int = static_cast<Annotation_list*>( list);
    return list_int->get_internal_annotation_list();
}

const MDL::IAnnotation_list* get_internal_annotation_list(
    const mi::neuraylib::IAnnotation_list* list)
{
    if( !list)
        return nullptr;
    const Annotation_list* list_int = static_cast<const Annotation_list*>( list);
    return list_int->get_internal_annotation_list();
}

template <class E, class I>
Expression_base<E, I>::~Expression_base() { }

template <class E, class I>
const mi::neuraylib::IType* Expression_base<E, I>::get_type() const
{
    mi::base::Handle<const MDL::IType> result_int( m_expr->get_type());
    mi::base::Handle<mi::neuraylib::IValue_factory> vf( m_ef->get_value_factory());
    mi::base::Handle<Type_factory> tf( static_cast<Type_factory*>( vf->get_type_factory()));
    return tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IValue* Expression_constant::get_value() const
{
    mi::base::Handle<const MDL::IValue> result_int( m_expr->get_value());
    mi::base::Handle<Value_factory> vf(  static_cast<Value_factory*>( m_ef->get_value_factory()));
    return vf->create( result_int.get(), m_owner.get());
}

mi::neuraylib::IValue* Expression_constant::get_value()
{
    mi::base::Handle<MDL::IValue> result_int( m_expr->get_value());
    mi::base::Handle<Value_factory> vf(  static_cast<Value_factory*>( m_ef->get_value_factory()));
    return vf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Expression_constant::set_value( mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> value_int( NEURAY::get_internal_value( value));
    return m_expr->set_value( value_int.get());
}

const char* Expression_call::get_call() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = m_expr->get_call();
    return db_transaction->tag_to_name( tag);
}

mi::Sint32 Expression_call::set_call( const char* name)
{
    if( !name)
        return -1;

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( name);
    if( !tag)
        return -2;

    mi::base::Handle<MDL::IExpression_factory> ef( m_ef->get_internal_expression_factory());
    mi::base::Handle<MDL::IValue_factory> vf( ef->get_value_factory());
    mi::base::Handle<MDL::IType_factory> tf( vf->get_type_factory());

    SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
    mi::base::Handle<const MDL::IType> actual_type;
    if( class_id == MDL::Mdl_function_call::id) {
        DB::Access<MDL::Mdl_function_call> call( tag, db_transaction);
        if (call->is_immutable())
            return -5; // prevent user-calls to default functions
        MDL::Execution_context context;
        if (!call->is_valid(db_transaction, &context))
            return -6; // prevent user-calls to invalid functions
        actual_type = call->get_return_type();
    } else if( class_id == MDL::Mdl_material_instance::id) { //-V547 PVS
        DB::Access<MDL::Mdl_material_instance> m(tag, db_transaction);
        if (m->is_immutable())
            return -5; // prevent user-calls to default materials
        MDL::Execution_context context;
        if (!m->is_valid(db_transaction, &context))
            return -6; // prevent user-calls to invalid materials
        actual_type = tf->get_predefined_struct( MDL::IType_struct::SID_MATERIAL);
    } else
        return -3;

    mi::base::Handle<const MDL::IType> expected_type( m_expr->get_type());
    if( tf->compare( actual_type.get(), expected_type.get()) != 0)
        return -4;

    m_expr->set_call( tag);
    return 0;
}

const char* Expression_direct_call::get_definition() const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();
    DB::Tag tag = m_expr->get_definition(db_transaction);
    if (!tag.is_valid())
        return nullptr;
    return db_transaction->tag_to_name( tag);
}

const mi::neuraylib::IExpression_list* Expression_direct_call::get_arguments() const
{
    mi::base::Handle<const MDL::IExpression_list> result_int( m_expr->get_arguments());
    return m_ef->create_expression_list( result_int.get(), m_owner.get());
}

const mi::neuraylib::IExpression* Expression_list::get_expression( mi::Size index) const
{
    mi::base::Handle<const MDL::IExpression> result_int( m_expression_list->get_expression( index));
    return m_ef->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IExpression* Expression_list::get_expression( const char* name) const
{
    mi::base::Handle<const MDL::IExpression> result_int( m_expression_list->get_expression( name));
    return m_ef->create( result_int.get(), m_owner.get());
}

mi::Sint32 Expression_list::set_expression( mi::Size index, const mi::neuraylib::IExpression* expr)
{
    mi::base::Handle<const MDL::IExpression> expr_int( NEURAY::get_internal_expression( expr));
    return m_expression_list->set_expression( index, expr_int.get());
}

mi::Sint32 Expression_list::set_expression(
    const char* name, const mi::neuraylib::IExpression* expr)
{
    mi::base::Handle<const MDL::IExpression> expr_int( NEURAY::get_internal_expression( expr));
    return m_expression_list->set_expression( name, expr_int.get());
}

mi::Sint32 Expression_list::add_expression(
    const char* name, const mi::neuraylib::IExpression* expr)
{
    mi::base::Handle<const MDL::IExpression> expr_int( NEURAY::get_internal_expression( expr));
    return m_expression_list->add_expression( name, expr_int.get());
}

MDL::IExpression_list* Expression_list::get_internal_expression_list()
{
    m_expression_list->retain();
    return m_expression_list.get();
}

const MDL::IExpression_list* Expression_list::get_internal_expression_list() const
{
    m_expression_list->retain();
    return m_expression_list.get();
}

const char* Annotation_definition::get_module() const
{
    return m_anno_def->get_module();
}

const char* Annotation_definition::get_name() const
{
    return m_anno_def->get_name();
}

const char* Annotation_definition::get_mdl_module_name() const
{
    return m_anno_def->get_mdl_module_name();
}

const char* Annotation_definition::get_mdl_simple_name() const
{
    return m_anno_def->get_mdl_simple_name();
}

const char* Annotation_definition::get_mdl_parameter_type_name( Size index) const
{
    return m_anno_def->get_mdl_parameter_type_name( index);
}

mi::neuraylib::IAnnotation_definition::Semantics Annotation_definition::get_semantic() const
{
    return m_anno_def->get_semantic();
}

mi::Size Annotation_definition::get_parameter_count() const
{
    return m_anno_def->get_parameter_count();
}

const char* Annotation_definition::get_parameter_name(mi::Size index) const
{
    return m_anno_def->get_parameter_name(index);
}

mi::Size Annotation_definition::get_parameter_index(const char* name) const
{
    if (!name)
        return mi::Size(-1);
    return m_anno_def->get_parameter_index(name);
}

const mi::neuraylib::IType_list* Annotation_definition::get_parameter_types() const
{
    mi::base::Handle<const MDL::IType_list> result_int(m_anno_def->get_parameter_types());
    return m_tf->create_type_list(result_int.get(), m_owner.get());
}

const mi::neuraylib::IExpression_list* Annotation_definition::get_defaults() const
{
    mi::base::Handle<const MDL::IExpression_list> result_int(m_anno_def->get_defaults());
    return m_ef->create_expression_list(result_int.get(), m_owner.get());
}

bool Annotation_definition::is_exported() const
{
    return m_anno_def->is_exported();
}

const mi::neuraylib::IAnnotation_block* Annotation_definition::get_annotations() const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int(m_anno_def->get_annotations());
    return m_ef->create_annotation_block(result_int.get(), m_owner.get());
}

const mi::neuraylib::IAnnotation* Annotation_definition::create_annotation(
    const mi::neuraylib::IExpression_list* arguments) const
{
    mi::base::Handle<const MDL::IExpression_list> arguments_int(
        get_internal_expression_list(arguments));
    mi::base::Handle<const MDL::IAnnotation> result_int(
        m_anno_def->create_annotation(arguments_int.get()));
    return m_ef->create_annotation(result_int.get(), m_owner.get());
}

const mi::neuraylib::IExpression_list* Annotation::get_arguments() const
{
    mi::base::Handle<const MDL::IExpression_list> result_int( m_annotation->get_arguments());
    return m_ef->create_expression_list( result_int.get(), m_owner.get());
}

const mi::neuraylib::IAnnotation_definition* Annotation::get_definition() const
{
    mi::base::Handle<const MDL::IAnnotation_definition> result_int(
        m_annotation->get_definition(m_ef->get_db_transaction()));

    return m_ef->create_annotation_definition(result_int.get(), m_owner.get());
}

MDL::IAnnotation* Annotation::get_internal_annotation()
{
    m_annotation->retain();
    return m_annotation.get();
}

const MDL::IAnnotation* Annotation::get_internal_annotation() const
{
    m_annotation->retain();
    return m_annotation.get();
}

const mi::neuraylib::IAnnotation* Annotation_block::get_annotation( mi::Size index) const
{
    mi::base::Handle<const MDL::IAnnotation> result_int(
        m_annotation_block->get_annotation( index));
    return m_ef->create_annotation( result_int.get(), m_owner.get());
}

mi::Sint32 Annotation_block::set_annotation(
    mi::Size index, const mi::neuraylib::IAnnotation* annotation)
{
    mi::base::Handle<const MDL::IAnnotation> anno_int(
        NEURAY::get_internal_annotation( annotation));
    return m_annotation_block->set_annotation( index, anno_int.get());
}

mi::Sint32 Annotation_block::add_annotation( mi::neuraylib::IAnnotation* annotation)
{
    mi::base::Handle<const MDL::IAnnotation> anno_int(
        NEURAY::get_internal_annotation( annotation));
    return m_annotation_block->add_annotation( anno_int.get());
}

MDL::IAnnotation_block* Annotation_block::get_internal_annotation_block()
{
    m_annotation_block->retain();
    return m_annotation_block.get();
}

const MDL::IAnnotation_block* Annotation_block::get_internal_annotation_block() const
{
    m_annotation_block->retain();
    return m_annotation_block.get();
}

const mi::neuraylib::IAnnotation_block* Annotation_list::get_annotation_block(
    mi::Size index) const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int(
        m_annotation_list->get_annotation_block( index));
    return m_ef->create_annotation_block( result_int.get(), m_owner.get());
}

const mi::neuraylib::IAnnotation_block* Annotation_list::get_annotation_block(
    const char* name) const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int(
        m_annotation_list->get_annotation_block( name));
    return m_ef->create_annotation_block( result_int.get(), m_owner.get());
}

mi::Sint32 Annotation_list::set_annotation_block(
   mi::Size index, const mi::neuraylib::IAnnotation_block* block)
{
    mi::base::Handle<const MDL::IAnnotation_block> block_int(
        NEURAY::get_internal_annotation_block( block));
    return m_annotation_list->set_annotation_block( index, block_int.get());
}

mi::Sint32 Annotation_list::set_annotation_block(
    const char* name, const mi::neuraylib::IAnnotation_block* block)
{
    mi::base::Handle<const MDL::IAnnotation_block> block_int(
        NEURAY::get_internal_annotation_block( block));
    return m_annotation_list->set_annotation_block( name, block_int.get());
}

mi::Sint32 Annotation_list::add_annotation_block(
    const char* name, const mi::neuraylib::IAnnotation_block* block)
{
    mi::base::Handle<const MDL::IAnnotation_block> block_int(
        NEURAY::get_internal_annotation_block( block));
    return m_annotation_list->add_annotation_block( name, block_int.get());
}

MDL::IAnnotation_list* Annotation_list::get_internal_annotation_list()
{
    m_annotation_list->retain();
    return m_annotation_list.get();
}

const MDL::IAnnotation_list* Annotation_list::get_internal_annotation_list() const
{
    m_annotation_list->retain();
    return m_annotation_list.get();
}

mi::neuraylib::IExpression_constant* Expression_factory::create_constant(
    mi::neuraylib::IValue* value) const
{
    if( !value)
        return nullptr;

    mi::base::Handle<MDL::IValue> value_int( get_internal_value( value));
    mi::base::Handle<MDL::IExpression_constant> result_int(
        m_ef->create_constant( value_int.get()));
    return new Expression_constant( this, result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IExpression_constant* Expression_factory::create_constant(
    const mi::neuraylib::IValue* value) const
{
    if( !value)
        return nullptr;

    mi::base::Handle<const MDL::IValue> value_int( get_internal_value( value));
    mi::base::Handle<const MDL::IExpression_constant> result_int(
        m_ef->create_constant( value_int.get()));

    // The const_cast is safe here since we return a constant expression which does not give
    // access to a mutable value.
    return new Expression_constant(
        this, const_cast<MDL::IExpression_constant*>( result_int.get()), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_call* Expression_factory::create_call( const char* name) const
{
    if( !name)
        return nullptr;

    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( name);
    if( !tag)
        return nullptr;

    SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
    mi::base::Handle<const MDL::IType> type_int;
    if( class_id == MDL::Mdl_function_call::id) {
        DB::Access<MDL::Mdl_function_call> call( tag, db_transaction);
        if( call->is_immutable())
            return nullptr; // prevent user-calls to default functions
        MDL::Execution_context context;
        if (!call->is_valid(db_transaction, &context))
            return nullptr; // prevent user-calls to invalid functions
        type_int = call->get_return_type();
    } else if( class_id == MDL::Mdl_material_instance::id) { //-V547 PVS
        DB::Access<MDL::Mdl_material_instance> m(tag, db_transaction);
        if( m->is_immutable())
            return nullptr; // prevent user-calls to default materials
        MDL::Execution_context context;
        if (!m->is_valid(db_transaction, &context))
            return nullptr; // prevent user-calls to invalid materials
        mi::base::Handle<MDL::IValue_factory> vf( m_ef->get_value_factory());
        mi::base::Handle<MDL::IType_factory> tf( vf->get_type_factory());
        type_int = tf->get_predefined_struct( MDL::IType_struct::SID_MATERIAL);
    } else
        return nullptr;

    mi::base::Handle<MDL::IExpression_call> result_int(
        m_ef->create_call( type_int.get(), tag));
    return new Expression_call( this, m_transaction.get(), result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_parameter* Expression_factory::create_parameter(
    const mi::neuraylib::IType* type, mi::Size index) const
{
    if( !type)
        return nullptr;

    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    mi::base::Handle<MDL::IExpression_parameter> result_int(
        m_ef->create_parameter( type_int.get(), index));
    return new Expression_parameter( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_direct_call* Expression_factory::create_direct_call(
    const char* name, mi::neuraylib::IExpression_list* arguments) const
{
    if( !name || !arguments)
        return nullptr;

    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = db_transaction->name_to_tag( name);
    if( !tag)
        return nullptr;

    SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
    mi::base::Handle<const MDL::IType> type_int;
    DB::Tag module_tag;
    MDL::Mdl_ident def_ident;
    if( class_id == MDL::Mdl_function_definition::id) {
        DB::Access<MDL::Mdl_function_definition> def( tag, db_transaction);
        type_int = def->get_return_type();
        module_tag = def->get_module(db_transaction);
        def_ident = def->get_ident();
    } else if( class_id == MDL::Mdl_material_definition::id) { //-V547 PVS
        mi::base::Handle<MDL::IValue_factory> vf( m_ef->get_value_factory());
        mi::base::Handle<MDL::IType_factory> tf( vf->get_type_factory());
        type_int = tf->get_predefined_struct( MDL::IType_struct::SID_MATERIAL);

        DB::Access<MDL::Mdl_material_definition> def(tag, db_transaction);
        module_tag = def->get_module(db_transaction);
        def_ident = def->get_ident();
    } else
        return nullptr;

    // TODO Check arguments against the parameters of the definition, reject array constructor;
    // TODO but the function is currently not exposed in the API.
    ASSERT( M_SCENE, false);

    mi::base::Handle<MDL::IExpression_list> arguments_int(
        get_internal_expression_list( arguments));
    mi::base::Handle<MDL::IExpression_direct_call> result_int(
        m_ef->create_direct_call(
            type_int.get(), module_tag,
            MDL::Mdl_tag_ident(tag, def_ident), name , arguments_int.get()));

    return new Expression_direct_call( this, m_transaction.get(), result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_temporary* Expression_factory::create_temporary(
    const mi::neuraylib::IType* type, mi::Size index) const
{
    if( !type)
        return nullptr;

    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    mi::base::Handle<MDL::IExpression_temporary> result_int(
        m_ef->create_temporary( type_int.get(), index));
    return new Expression_temporary( this, result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_list* Expression_factory::create_expression_list() const
{
    mi::base::Handle<MDL::IExpression_list> result_int( m_ef->create_expression_list());
    return create_expression_list( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IAnnotation* Expression_factory::create_annotation(
    const char* name, const mi::neuraylib::IExpression_list* arguments) const
{
    mi::base::Handle<const MDL::IExpression_list> arguments_int(
        get_internal_expression_list( arguments));
    mi::base::Handle<MDL::IAnnotation> result_int(
        m_ef->create_annotation( name, arguments_int.get()));
    return create_annotation( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IAnnotation_block* Expression_factory::create_annotation_block() const
{
    mi::base::Handle<MDL::IAnnotation_block> result_int(
        m_ef->create_annotation_block());
    return create_annotation_block( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IAnnotation_list* Expression_factory::create_annotation_list() const
{
    mi::base::Handle<MDL::IAnnotation_list> result_int(
        m_ef->create_annotation_list());
    return create_annotation_list( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression* Expression_factory::clone( const mi::neuraylib::IExpression* expr) const
{
    mi::base::Handle<const MDL::IExpression> expr_int( get_internal_expression( expr));
    DB::Transaction* db_transaction = get_db_transaction();
    mi::base::Handle<MDL::IExpression> result_int( m_ef->clone(
        expr_int.get(), db_transaction, true));
    return create( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IExpression_list* Expression_factory::clone(
   const mi::neuraylib::IExpression_list* expr_list) const
{
    mi::base::Handle<const MDL::IExpression_list> expr_list_int(
       get_internal_expression_list( expr_list));
    DB::Transaction* db_transaction = get_db_transaction();
    mi::base::Handle<MDL::IExpression_list> result_int( m_ef->clone(
        expr_list_int.get(), db_transaction, true));
    return create_expression_list( result_int.get(), /*owner*/ nullptr);
}

mi::Sint32 Expression_factory::compare(
    const mi::neuraylib::IExpression* lhs, const mi::neuraylib::IExpression* rhs) const
{
    mi::base::Handle<const MDL::IExpression> lhs_int( get_internal_expression( lhs));
    mi::base::Handle<const MDL::IExpression> rhs_int( get_internal_expression( rhs));
    return m_ef->compare( lhs_int.get(), rhs_int.get());
}

mi::Sint32 Expression_factory::compare(
    const mi::neuraylib::IExpression_list* lhs, const mi::neuraylib::IExpression_list* rhs) const
{
    mi::base::Handle<const MDL::IExpression_list> lhs_int( get_internal_expression_list( lhs));
    mi::base::Handle<const MDL::IExpression_list> rhs_int( get_internal_expression_list( rhs));
    return m_ef->compare( lhs_int.get(), rhs_int.get());
}

const mi::IString* Expression_factory::dump(
    const mi::neuraylib::IExpression* expr, const char* name, mi::Size depth) const
{
    if( !expr)
        return nullptr;

    mi::base::Handle<const MDL::IExpression> expr_int( get_internal_expression( expr));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_ef->dump( db_transaction, expr_int.get(), name, depth);
}

const mi::IString* Expression_factory::dump(
    const mi::neuraylib::IExpression_list* list, const char* name, mi::Size depth) const
{
    if( !list)
        return nullptr;

    mi::base::Handle<const MDL::IExpression_list> list_int( get_internal_expression_list( list));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_ef->dump( db_transaction, list_int.get(), name, depth);
}

const mi::IString* Expression_factory::dump(
    const mi::neuraylib::IAnnotation* annotation, const char* name, mi::Size depth) const
{
    if( !annotation)
        return nullptr;

    mi::base::Handle<const MDL::IAnnotation> anno_int( get_internal_annotation( annotation));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_ef->dump( db_transaction, anno_int.get(), name, depth);
}

const mi::IString* Expression_factory::dump(
    const mi::neuraylib::IAnnotation_block* block, const char* name, mi::Size depth) const
{
    if( !block)
        return nullptr;

    mi::base::Handle<const MDL::IAnnotation_block> block_int( get_internal_annotation_block(block));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_ef->dump( db_transaction, block_int.get(), name, depth);
}

const mi::IString* Expression_factory::dump(
    const mi::neuraylib::IAnnotation_list* list, const char* name, mi::Size depth) const
{
    if( !list)
        return nullptr;

    mi::base::Handle<const MDL::IAnnotation_list> list_int( get_internal_annotation_list( list));
    DB::Transaction* db_transaction = get_db_transaction();
    return m_ef->dump( db_transaction, list_int.get(), name, depth);
}

mi::neuraylib::IExpression* Expression_factory::create(
    MDL::IExpression* expr, const mi::base::IInterface* owner) const
{
    if( !expr)
        return nullptr;

    MDL::IExpression::Kind kind = expr->get_kind();

    switch( kind) {
        case MDL::IExpression::EK_CONSTANT: {
            mi::base::Handle<MDL::IExpression_constant> e(
                expr->get_interface<MDL::IExpression_constant>());
            return new Expression_constant( this, e.get(), owner);
        }
        case MDL::IExpression::EK_CALL: {
            mi::base::Handle<MDL::IExpression_call> e(
                expr->get_interface<MDL::IExpression_call>());
            return new Expression_call( this, m_transaction.get(), e.get(), owner);
        }
        case MDL::IExpression::EK_PARAMETER: {
            mi::base::Handle<MDL::IExpression_parameter> e(
                expr->get_interface<MDL::IExpression_parameter>());
            return new Expression_parameter( this, e.get(), owner);
        }
        case MDL::IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<MDL::IExpression_direct_call> e(
                expr->get_interface<MDL::IExpression_direct_call>());
            return new Expression_direct_call( this, m_transaction.get(), e.get(), owner);
        }
        case MDL::IExpression::EK_TEMPORARY: {
            mi::base::Handle<MDL::IExpression_temporary> e(
                expr->get_interface<MDL::IExpression_temporary>());
            return new Expression_temporary( this, e.get(), owner);
        }
        case MDL::IExpression::EK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    };

    ASSERT( M_SCENE, false);
    return nullptr;
}

const mi::neuraylib::IExpression* Expression_factory::create(
    const MDL::IExpression* expr, const mi::base::IInterface* owner) const
{
    return create( const_cast<MDL::IExpression*>( expr), owner);
}

mi::neuraylib::IExpression_list* Expression_factory::create_expression_list(
    MDL::IExpression_list* expr_list, const mi::base::IInterface* owner) const
{
    return expr_list ? new Expression_list( this, expr_list, owner) : nullptr;
}

const mi::neuraylib::IExpression_list* Expression_factory::create_expression_list(
    const MDL::IExpression_list* expr_list, const mi::base::IInterface* owner) const
{
    return create_expression_list( const_cast<MDL::IExpression_list*>( expr_list), owner);
}

mi::neuraylib::IAnnotation* Expression_factory::create_annotation(
    MDL::IAnnotation* anno, const mi::base::IInterface* owner) const
{
    return anno ? new Annotation( this, anno, owner) : nullptr;
}

const mi::neuraylib::IAnnotation* Expression_factory::create_annotation( //-V659 PVS
    const MDL::IAnnotation* anno, const mi::base::IInterface* owner) const
{
    return create_annotation( const_cast<MDL::IAnnotation*>( anno), owner);
}

mi::neuraylib::IAnnotation_block* Expression_factory::create_annotation_block(
    MDL::IAnnotation_block* block, const mi::base::IInterface* owner) const
{
    return block ? new Annotation_block( this, block, owner) : nullptr;
}

const mi::neuraylib::IAnnotation_block* Expression_factory::create_annotation_block(
    const MDL::IAnnotation_block* block, const mi::base::IInterface* owner) const
{
    return create_annotation_block( const_cast<MDL::IAnnotation_block*>( block), owner);
}

mi::neuraylib::IAnnotation_list* Expression_factory::create_annotation_list(
    MDL::IAnnotation_list* list, const mi::base::IInterface* owner) const
{
    return list ? new Annotation_list( this, list, owner) : nullptr;
}

const mi::neuraylib::IAnnotation_list* Expression_factory::create_annotation_list(
    const MDL::IAnnotation_list* list, const mi::base::IInterface* owner) const
{
    return create_annotation_list( const_cast<MDL::IAnnotation_list*>( list), owner);
}

const mi::neuraylib::IAnnotation_definition* Expression_factory::create_annotation_definition(
    const MDL::IAnnotation_definition* anno_def, const mi::base::IInterface* owner) const
{
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>(m_transaction.get());
    mi::base::Handle<Type_factory> tf(transaction_impl->get_type_factory());

    return anno_def ? new Annotation_definition(this, tf.get(), anno_def, owner) : nullptr;
}

mi::neuraylib::IExpression* Expression_factory::create_cast(
    mi::neuraylib::IExpression* src_expr,
    const mi::neuraylib::IType* target_type,
    const char* cast_db_name,
    bool force_cast,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_errors;
    if (errors == nullptr)
        errors = &dummy_errors;
    *errors = 0;

    if (!(src_expr && target_type)) {
        *errors = -1;
        return nullptr;
    }
    mi::base::Handle<MDL::IExpression> src_expr_int(get_internal_expression(src_expr));
    mi::base::Handle<const MDL::IType> target_type_int(get_internal_type(target_type));

    mi::base::Handle<MDL::IExpression> result_int(m_ef->create_cast(
        get_db_transaction(),
        src_expr_int.get(),
        target_type_int.get(),
        cast_db_name,
        force_cast,
        /*direct_call=*/false,
        errors));

    if (src_expr_int == result_int) {
        src_expr->retain();
        return src_expr;
    }
    return create(result_int.get(), /*owner=*/ nullptr);
}

DB::Transaction* Expression_factory::get_db_transaction() const
{
    if( !m_transaction)
        return nullptr;

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( m_transaction.get());
    return transaction_impl->get_db_transaction();
}

} // namespace NEURAY

} // namespace MI
