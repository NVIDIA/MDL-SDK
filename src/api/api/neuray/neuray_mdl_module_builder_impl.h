/***************************************************************************************************
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMdl_module_builder implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MDL_MODULE_BUILDER_IMPL_H
#define API_API_NEURAY_NEURAY_MDL_MODULE_BUILDER_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_module_builder.h>
#include <mi/neuraylib/imodule.h>

#include <memory>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace DB { class Transaction; }
namespace MDL { class Mdl_module_builder; }

namespace NEURAY {

class Mdl_module_builder_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_module_builder>
{
public:
    Mdl_module_builder_impl(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        mi::neuraylib::Mdl_version min_module_version,
        mi::neuraylib::Mdl_version max_module_version,
        mi::neuraylib::IMdl_execution_context* context);

    Mdl_module_builder_impl( const Mdl_module_builder_impl& ) = delete;
    Mdl_module_builder_impl& operator=( const Mdl_module_builder_impl& ) = delete;

    ~Mdl_module_builder_impl();

    // public API methods

    mi::Sint32 add_variant(
        const char* name,
        const char* prototype_name,
        const mi::neuraylib::IExpression_list* defaults,
        const mi::neuraylib::IAnnotation_block* annotations,
        const mi::neuraylib::IAnnotation_block* return_annotations,
        bool is_exported,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 add_function(
        const char* name,
        const mi::neuraylib::IExpression* body,
        const mi::neuraylib::IType_list* parameters,
        const mi::neuraylib::IExpression_list* defaults,
        const mi::neuraylib::IAnnotation_list* parameter_annotations,
        const mi::neuraylib::IAnnotation_block* annotations,
        const mi::neuraylib::IAnnotation_block* return_annotations,
        bool is_exported,
        mi::neuraylib::IType::Modifier frequency_qualifier,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 add_annotation(
        const char* name,
        const mi::neuraylib::IType_list* parameters,
        const mi::neuraylib::IExpression_list* defaults,
        const mi::neuraylib::IAnnotation_list* parameter_annotations,
        const mi::neuraylib::IAnnotation_block* annotations,
        bool is_exported,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 add_enum_type(
        const char* name,
        const mi::neuraylib::IExpression_list* enumerators,
        const mi::neuraylib::IAnnotation_list* enumerator_annotations,
        const mi::neuraylib::IAnnotation_block* annotations,
        bool is_exported,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 add_struct_type(
        const char* name,
        const mi::neuraylib::IType_list* fields,
        const mi::neuraylib::IExpression_list* field_defaults,
        const mi::neuraylib::IAnnotation_list* field_annotations,
        const mi::neuraylib::IAnnotation_block* annotations,
        bool is_exported,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 add_constant(
        const char* name,
        const mi::neuraylib::IExpression* expr,
        const mi::neuraylib::IAnnotation_block* annotations,
        bool is_exported,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 set_module_annotations(
        const mi::neuraylib::IAnnotation_block* annotations,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 remove_entity(
        const char* name,
        mi::Size index,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 clear_module(
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::IArray* analyze_uniform(
        const mi::neuraylib::IExpression* root_expr,
        bool root_expr_uniform,
        mi::neuraylib::IMdl_execution_context* context) final;

private:
    DB::Transaction* m_db_transaction;
    std::unique_ptr<MDL::Mdl_module_builder> m_impl;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_MODULE_BUILDER_IMPL_H
