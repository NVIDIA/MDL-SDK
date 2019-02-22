/***************************************************************************************************
 * Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <boost/scoped_array.hpp>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/iattribute_container.h>
#include <mi/neuraylib/imaterial_definition.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure.h>
#include <api/api/neuray/neuray_mdl_execution_context_impl.h>
#include <base/lib/log/i_log_logger.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>

#include "neuray_attribute_container_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_class_factory.h"

#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include "neuray_scope_impl.h"

namespace MI {

namespace NEURAY {

Mdl_factory_impl::Mdl_factory_impl(
    mi::neuraylib::INeuray* neuray, const Class_factory* class_factory)
  : m_neuray( neuray), m_class_factory( class_factory)
{
}

Mdl_factory_impl::~Mdl_factory_impl()
{
    m_class_factory = 0;
    m_neuray = 0;
}

mi::neuraylib::IType_factory* Mdl_factory_impl::create_type_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_type_factory( transaction) : 0;
}

mi::neuraylib::IValue_factory* Mdl_factory_impl::create_value_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_value_factory( transaction) : 0;
}

mi::neuraylib::IExpression_factory* Mdl_factory_impl::create_expression_factory(
    mi::neuraylib::ITransaction* transaction)
{
    return transaction ? m_class_factory->create_expression_factory( transaction) : 0;
}

mi::Sint32 Mdl_factory_impl::create_variants(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const mi::IArray* variant_data)
{
    if( !module_name || !variant_data)
        return -5;
    mi::Size variant_count = variant_data->get_length();
    if( variant_count == 0)
        return -5;

    if( !transaction)
        return -5;
    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    boost::scoped_array<MDL::Variant_data> mdl_variant_data( new MDL::Variant_data[variant_count]);
    for( mi::Size i = 0; i < variant_count; ++i) {

        mi::base::Handle<const mi::IStructure> variant(
            variant_data->get_value<mi::IStructure>( i));
        if( !variant)
            return -5;

        mi::base::Handle<const mi::IString> variant_name(
            variant->get_value<mi::IString>( "variant_name"));
        if( !variant_name)
            variant_name = variant->get_value<mi::IString>( "preset_name");
        if( !variant_name)
            return -5;
        mdl_variant_data[i].m_variant_name = variant_name->get_c_str();

        mi::base::Handle<const mi::IString> prototype_name(
            variant->get_value<mi::IString>( "prototype_name"));
        if( !prototype_name)
            return -5;
        DB::Tag tag = db_transaction->name_to_tag( prototype_name->get_c_str());
        if( !tag)
            return -5;
        SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
        if(    class_id != MDL::ID_MDL_MATERIAL_DEFINITION
            && class_id != MDL::ID_MDL_FUNCTION_DEFINITION)
            return -5;
        mdl_variant_data[i].m_prototype_tag = tag;

        mi::base::Handle<const mi::neuraylib::IExpression_list> defaults(
            variant->get_value<mi::neuraylib::IExpression_list>( "defaults"));
        mdl_variant_data[i].m_defaults = get_internal_expression_list( defaults.get());

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annotations(
            variant->get_value<mi::neuraylib::IAnnotation_block>( "annotations"));
        mdl_variant_data[i].m_annotations = get_internal_annotation_block( annotations.get());
    }

    MDL::Execution_context context;
    return MDL::Mdl_module::create_module(
        db_transaction, module_name, mdl_variant_data.get(), variant_count, &context);
}

mi::Sint32 Mdl_factory_impl::create_materials(
    mi::neuraylib::ITransaction* transaction,
    const char* module_name,
    const mi::IArray* material_data)
{
    return 0;
}

mi::neuraylib::IValue_texture* Mdl_factory_impl::create_texture(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    mi::neuraylib::IType_texture::Shape shape,
    mi::Float32 gamma,
    bool shared,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction) {
        *errors = -1;
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    MDL::IType_texture::Shape int_shape = ext_shape_to_int_shape( shape);
    mi::base::Handle<MDL::IValue_texture> result( MDL::Mdl_module::create_texture(
        db_transaction, file_path, int_shape, gamma, shared, errors));
    if( !result)
        return 0;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_texture>( result.get(), /*owner*/ 0);
}

mi::neuraylib::IValue_light_profile* Mdl_factory_impl::create_light_profile(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    bool shared,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction) {
        *errors = -1;
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<MDL::IValue_light_profile> result(
        MDL::Mdl_module::create_light_profile( db_transaction, file_path, shared, errors));
    if( !result)
        return 0;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_light_profile>( result.get(), /*owner*/ 0);
}

mi::neuraylib::IValue_bsdf_measurement* Mdl_factory_impl::create_bsdf_measurement(
    mi::neuraylib::ITransaction* transaction,
    const char* file_path,
    bool shared,
    mi::Sint32* errors)
{
    mi::Sint32 dummy_errors = 0;
    if( !errors)
        errors = &dummy_errors;

    if( !transaction) {
        *errors = -1;
        return 0;
    }

    Transaction_impl* transaction_impl = static_cast<Transaction_impl*>( transaction);
    DB::Transaction* db_transaction = transaction_impl->get_db_transaction();

    mi::base::Handle<MDL::IValue_bsdf_measurement> result(
        MDL::Mdl_module::create_bsdf_measurement( db_transaction, file_path, shared, errors));
    if( !result)
        return 0;

    mi::base::Handle<Value_factory> vf( transaction_impl->get_value_factory());
    return vf->create<mi::neuraylib::IValue_bsdf_measurement>( result.get(), /*owner*/ 0);
}

mi::neuraylib::IMdl_execution_context* Mdl_factory_impl::create_execution_context()
{
    return new Mdl_execution_context_impl();
}

} // namespace NEURAY

} // namespace MI

