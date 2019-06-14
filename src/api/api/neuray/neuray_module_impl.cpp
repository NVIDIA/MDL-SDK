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

/** \file
 ** \brief Source for the IModule implementation.
 **/

#include "pch.h"

#include "neuray_expression_impl.h"
#include "neuray_module_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/istring.h>
#include <base/system/main/access_module.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Module_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return 0;
    if( argc != 0)
        return 0;
    return new MDL::Mdl_module;
}

mi::base::IInterface* Module_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return 0;
    if( argc != 0)
        return 0;
    return (new Module_impl())->cast_to_major();
}

mi::neuraylib::Element_type Module_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_MODULE;
}

const char* Module_impl::get_filename() const
{
    return get_db_element()->get_api_filename();
}

const char* Module_impl::get_mdl_name() const
{
    return get_db_element()->get_mdl_name();
}

mi::Size Module_impl::get_import_count() const
{
    return get_db_element()->get_import_count();
}

const char* Module_impl::get_import( mi::Size index) const
{
    DB::Tag tag = get_db_element()->get_import( index);
    return get_db_transaction()->tag_to_name( tag);
}

const mi::neuraylib::IValue_list* Module_impl::get_constants() const
{
    mi::base::Handle<Value_factory> vf( get_transaction()->get_value_factory());
    mi::base::Handle<const MDL::IValue_list> result_int( get_db_element()->get_constants());
    return vf->create_value_list( result_int.get(), this->cast_to_major());
}

const mi::neuraylib::IType_list* Module_impl::get_types() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IType_list> result_int( get_db_element()->get_types());
    return tf->create_type_list( result_int.get(), this->cast_to_major());
}

mi::Size Module_impl::get_function_count() const
{
    return get_db_element()->get_function_count();
}

const char* Module_impl::get_function( mi::Size index) const
{
    DB::Transaction* db_transaction = get_db_transaction();
    return get_db_element()->get_function_name(db_transaction, index);
}

mi::Size Module_impl::get_material_count() const
{
    return get_db_element()->get_material_count();
}

const char* Module_impl::get_material( mi::Size index) const
{
    DB::Transaction* db_transaction = get_db_transaction();
    return get_db_element()->get_material_name(db_transaction, index);
}

const mi::neuraylib::IAnnotation_block* Module_impl::get_annotations() const
{
    mi::base::Handle<Expression_factory> ef( get_transaction()->get_expression_factory());
    mi::base::Handle<const MDL::IAnnotation_block> result_int(
        get_db_element()->get_annotations());
    return ef->create_annotation_block( result_int.get(), this->cast_to_major());
}

bool Module_impl::is_standard_module() const
{
    return get_db_element()->is_standard_module();
}

const mi::IArray* Module_impl::get_function_overloads(
    const char* name, const mi::neuraylib::IExpression_list* arguments) const
{
    if( !name)
        return 0;

    mi::base::Handle<const MDL::IExpression_list> arguments_int(
        get_internal_expression_list( arguments));

    const std::vector<std::string>& tmp = get_db_element()->get_function_overloads(
        get_db_transaction(), name, arguments_int.get());

    mi::base::Handle<mi::IDynamic_array> result(
        get_transaction()->create<mi::IDynamic_array>( "String[]"));
    result->set_length( tmp.size());

    for( mi::Size i = 0, n = tmp.size(); i < n; ++i) {
        mi::base::Handle<mi::IString> element( result->get_element<mi::IString>( i));
        element->set_c_str( tmp[i].c_str());
    }

    result->retain();
    return result.get();
}

const mi::IArray* Module_impl::get_function_overloads(
    const char* name, const char* param_sig) const
{
    if( !name)
        return 0;

    const std::vector<std::string>& tmp =
        get_db_element()->get_function_overloads_by_signature(
            get_db_transaction(), name, param_sig);

    mi::base::Handle<mi::IDynamic_array> result(
        get_transaction()->create<mi::IDynamic_array>( "String[]"));
    result->set_length( tmp.size());

    for( mi::Size i = 0, n = tmp.size(); i < n; ++i) {
        mi::base::Handle<mi::IString> element( result->get_element<mi::IString>( i));
        element->set_c_str( tmp[i].c_str());
    }

    result->retain();
    return result.get();
}

mi::Size Module_impl::get_resources_count() const
{
    return get_db_element()->get_resources_count();
}

const mi::neuraylib::IType_resource* Module_impl::get_resource_type(mi::Size index) const
{
    mi::base::Handle<const MI::MDL::IType_resource> int_resource_type(
        get_db_element()->get_resource_type(index));

    mi::base::Handle<Type_factory> tf(get_transaction()->get_type_factory());
    return tf->create<mi::neuraylib::IType_resource>(
        int_resource_type.get(), this->cast_to_major());
}

const char* Module_impl::get_resource_mdl_file_path(mi::Size index) const
{
    return get_db_element()->get_resource_mdl_file_path(index);
}


const char* Module_impl::get_resource_name(mi::Size index) const
{
    DB::Tag tag(get_db_element()->get_resource_tag(index));
    return get_db_transaction()->tag_to_name(tag);
}

mi::Size Module_impl::get_annotation_definition_count() const
{
    return get_db_element()->get_annotation_definition_count();
}

const mi::neuraylib::IAnnotation_definition* Module_impl::get_annotation_definition(
    mi::Size index) const
{
    mi::base::Handle<const MDL::IAnnotation_definition> result_int(
        get_db_element()->get_annotation_definition(index));
    mi::base::Handle<Expression_factory> ef(get_transaction()->get_expression_factory());
    return ef->create_annotation_definition(result_int.get(), this->cast_to_major());

}

const mi::neuraylib::IAnnotation_definition* Module_impl::get_annotation_definition(
    const char *name) const
{
    mi::base::Handle<const MDL::IAnnotation_definition> result_int(
        get_db_element()->get_annotation_definition(name));
    mi::base::Handle<Expression_factory> ef(get_transaction()->get_expression_factory());
    return ef->create_annotation_definition(result_int.get(), this->cast_to_major());
}

} // namespace NEURAY

} // namespace MI
