/***************************************************************************************************
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IMdl_module_builder implementation.
 **/

#include "pch.h"

#include "neuray_mdl_module_builder_impl.h"

#include <mi/neuraylib/inumber.h>
#include <base/data/db/i_db_transaction.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/mdl_elements/i_mdl_elements_module_builder.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/mdl_elements_utilities.h>

#include "neuray_array_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_value_impl.h"

namespace MI {

namespace NEURAY {

Mdl_module_builder_impl::Mdl_module_builder_impl(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    mi::neuraylib::Mdl_version min_module_version,
    mi::neuraylib::Mdl_version max_module_version,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    Transaction_impl* transaction_impl
        = static_cast<Transaction_impl*>( transaction);
    m_db_transaction = transaction_impl->get_db_transaction();
    m_db_transaction->pin();

    mi::mdl::IMDL::MDL_version min_mdl_module_version
        = MDL::convert_mdl_version( min_module_version);
    mi::mdl::IMDL::MDL_version max_mdl_module_version
        = MDL::convert_mdl_version( max_module_version);

    m_impl.reset( new MDL::Mdl_module_builder(
        m_db_transaction,
        module_name,
        min_mdl_module_version,
        max_mdl_module_version,
        /*export_to_db*/ true,
        context_impl));
}

Mdl_module_builder_impl::~Mdl_module_builder_impl()
{
    m_db_transaction->unpin();
}

mi::Sint32 Mdl_module_builder_impl::add_variant(
    const char* name,
    const char* prototype_name,
    const mi::neuraylib::IExpression_list* defaults,
    const mi::neuraylib::IAnnotation_block* annotations,
    const mi::neuraylib::IAnnotation_block* return_annotations,
    bool is_exported,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name || !prototype_name) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    DB::Tag tag = m_db_transaction->name_to_tag( prototype_name);
    if( !tag) {
        add_error_message( context_impl,
            STRING::formatted_string( "Invalid prototype name \"%s\".", prototype_name), -10);
        return -1;
    }

    mi::base::Handle<const MDL::IExpression_list> int_defaults(
        get_internal_expression_list( defaults));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_return_annotations(
        get_internal_annotation_block( return_annotations));

    return m_impl->add_variant(
        name,
        tag,
        int_defaults.get(),
        int_annotations.get(),
        int_return_annotations.get(),
        is_exported,
        context_impl);
}

mi::Sint32 Mdl_module_builder_impl::add_function(
    const char* name,
    const mi::neuraylib::IExpression* body,
    const mi::neuraylib::IType_list* parameters,
    const mi::neuraylib::IExpression_list* defaults,
    const mi::neuraylib::IAnnotation_list* parameter_annotations,
    const mi::neuraylib::IAnnotation_block* annotations,
    const mi::neuraylib::IAnnotation_block* return_annotations,
    bool is_exported,
    mi::neuraylib::IType::Modifier frequency_qualifier,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name || !body) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    mi::base::Handle<const MDL::IExpression> int_body(
        get_internal_expression( body));
    mi::base::Handle<const MDL::IType_list> int_parameters(
        get_internal_type_list( parameters));
    mi::base::Handle<const MDL::IExpression_list> int_defaults(
        get_internal_expression_list( defaults));
    mi::base::Handle<const MDL::IAnnotation_list> int_parameter_annotations(
        get_internal_annotation_list( parameter_annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_return_annotations(
        get_internal_annotation_block( return_annotations));
    MDL::IType::Modifier int_frequency_modifier
        = static_cast<MDL::IType::Modifier>( ext_modifiers_to_int_modifiers( frequency_qualifier));

    return m_impl->add_function(
        name,
        int_body.get(),
        int_parameters.get(),
        int_defaults.get(),
        int_parameter_annotations.get(),
        int_annotations.get(),
        int_return_annotations.get(),
        is_exported,
        int_frequency_modifier,
        context_impl);
}

mi::Sint32 Mdl_module_builder_impl::add_annotation(
    const char* name,
    const mi::neuraylib::IType_list* parameters,
    const mi::neuraylib::IExpression_list* defaults,
    const mi::neuraylib::IAnnotation_list* parameter_annotations,
    const mi::neuraylib::IAnnotation_block* annotations,
    bool is_exported,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    mi::base::Handle<const MDL::IType_list> int_parameters(
        get_internal_type_list( parameters));
    mi::base::Handle<const MDL::IExpression_list> int_defaults(
        get_internal_expression_list( defaults));
    mi::base::Handle<const MDL::IAnnotation_list> int_parameter_annotations(
        get_internal_annotation_list( parameter_annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));

    return m_impl->add_annotation(
        name,
        int_parameters.get(),
        int_defaults.get(),
        int_parameter_annotations.get(),
        int_annotations.get(),
        is_exported,
        context_impl);
}

mi::Sint32 Mdl_module_builder_impl::add_enum_type(
    const char* name,
    const mi::neuraylib::IExpression_list* enumerators,
    const mi::neuraylib::IAnnotation_list* enumerator_annotations,
    const mi::neuraylib::IAnnotation_block* annotations,
    bool is_exported,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name || !enumerators) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    mi::base::Handle<const MDL::IExpression_list> int_enumerators(
        get_internal_expression_list( enumerators));
    mi::base::Handle<const MDL::IAnnotation_list> int_enumerator_annotations(
        get_internal_annotation_list( enumerator_annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));

    return m_impl->add_enum_type(
        name,
        int_enumerators.get(),
        int_enumerator_annotations.get(),
        int_annotations.get(),
        is_exported,
        context_impl);
}

mi::Sint32 Mdl_module_builder_impl::add_struct_type(
    const char* name,
    const mi::neuraylib::IType_list* fields,
    const mi::neuraylib::IExpression_list* field_defaults,
    const mi::neuraylib::IAnnotation_list* field_annotations,
    const mi::neuraylib::IAnnotation_block* annotations,
    bool is_exported,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name || !fields) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    mi::base::Handle<const MDL::IType_list> int_fields(
        get_internal_type_list( fields));
    mi::base::Handle<const MDL::IExpression_list> int_field_defaults(
        get_internal_expression_list( field_defaults));
    mi::base::Handle<const MDL::IAnnotation_list> int_field_annotations(
        get_internal_annotation_list( field_annotations));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));

    return m_impl->add_struct_type(
        name,
        int_fields.get(),
        int_field_defaults.get(),
        int_field_annotations.get(),
        int_annotations.get(),
        is_exported,
        context_impl);
}

mi::Sint32 Mdl_module_builder_impl::add_constant(
    const char* name,
    const mi::neuraylib::IExpression* expr,
    const mi::neuraylib::IAnnotation_block* annotations,
    bool is_exported,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name || !expr) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    mi::base::Handle<const MDL::IExpression> int_expr(
        get_internal_expression( expr));
    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));

    return m_impl->add_constant(
        name, int_expr.get(), int_annotations.get(), is_exported, context_impl);
}

mi::Sint32 Mdl_module_builder_impl::set_module_annotations(
    const mi::neuraylib::IAnnotation_block* annotations,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    mi::base::Handle<const MDL::IAnnotation_block> int_annotations(
        get_internal_annotation_block( annotations));

    return m_impl->set_module_annotations( int_annotations.get(), context_impl);
}

mi::Sint32 Mdl_module_builder_impl::remove_entity(
    const char* name,
    mi::Size index,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !name) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }

    return m_impl->remove_entity( name, index, context_impl);
}

mi::Sint32 Mdl_module_builder_impl::clear_module(
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    return m_impl->clear_module( context_impl);
}

const mi::IArray* Mdl_module_builder_impl::analyze_uniform(
    const mi::neuraylib::IExpression* root_expr,
    bool root_expr_uniform,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* context_impl = unwrap_and_clear_context( context, default_context);

    if( !root_expr || !context) {
        add_error_message( context_impl, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    mi::base::Handle<const MDL::IExpression> int_root_expr(
        get_internal_expression( root_expr));

    std::vector<bool> result = m_impl->analyze_uniform(
        int_root_expr.get(), root_expr_uniform, context_impl);
    if( context_impl->get_error_messages_count() > 0)
        return nullptr;

    mi::Size n = result.size();
    mi::base::Handle<mi::IArray> array( new Array_impl( nullptr, "Boolean", n));
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<mi::IBoolean> element( array->get_element<mi::IBoolean>( i));
        element->set_value( result[i]);
    }

    array->retain();
    return array.get();
}

} // namespace NEURAY

} // namespace MI
