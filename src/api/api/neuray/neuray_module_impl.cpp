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

/** \file
 ** \brief Source for the IModule implementation.
 **/

#include "pch.h"

#include "neuray_expression_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_module_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/istring.h>

#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/system/main/access_module.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Module_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new MDL::Mdl_module;
}

mi::base::IInterface* Module_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
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

mi::Size Module_impl::get_mdl_package_component_count() const
{
    return get_db_element()->get_mdl_package_component_count();
}

const char* Module_impl::get_mdl_package_component_name( mi::Size index) const
{
    return get_db_element()->get_mdl_package_component_name( index);
}

const char* Module_impl::get_mdl_simple_name() const
{
    return get_db_element()->get_mdl_simple_name();
}

mi::neuraylib::Mdl_version Module_impl::get_mdl_version() const
{
    return get_db_element()->get_mdl_version();
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

const mi::neuraylib::IStruct_category_list* Module_impl::get_struct_categories() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IStruct_category_list> result_int(
        get_db_element()->get_struct_categories());
    return tf->create_struct_category_list( result_int.get(), this->cast_to_major());
}

const mi::neuraylib::IType_list* Module_impl::get_types() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IType_list> result_int( get_db_element()->get_types());
    return tf->create_type_list( result_int.get(), this->cast_to_major());
}

const mi::neuraylib::IValue_list* Module_impl::get_constants() const
{
    mi::base::Handle<Value_factory> vf( get_transaction()->get_value_factory());
    mi::base::Handle<const MDL::IValue_list> result_int( get_db_element()->get_constants());
    return vf->create_value_list( result_int.get(), this->cast_to_major());
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

bool Module_impl::is_mdle_module() const
{
    return get_db_element()->is_mdle_module();
}

const mi::IArray* Module_impl::get_function_overloads(
    const char* name, const mi::neuraylib::IExpression_list* arguments) const
{
    if( !name)
        return nullptr;

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

    return result.extract();
}

const mi::IArray* Module_impl::get_function_overloads(
    const char* name, const mi::IArray* parameter_types) const
{
    if( !name)
        return nullptr;

    std::vector<const char*> paramter_types_vector;
    for( size_t i = 0, n = parameter_types ? parameter_types->get_length() : 0; i < n; ++i) {
        mi::base::Handle<const mi::IString> element( parameter_types->get_element<mi::IString>( i));
        if( !element)
            return nullptr;
        paramter_types_vector.push_back( element->get_c_str());
    }

    const std::vector<std::string>& tmp
        = get_db_element()->get_function_overloads_by_signature(
            name, paramter_types_vector);

    mi::base::Handle<mi::IDynamic_array> result(
        get_transaction()->create<mi::IDynamic_array>( "String[]"));
    result->set_length( tmp.size());

    for( mi::Size i = 0, n = tmp.size(); i < n; ++i) {
        mi::base::Handle<mi::IString> element( result->get_element<mi::IString>( i));
        element->set_c_str( tmp[i].c_str());
    }

    return result.extract();
}

mi::Size Module_impl::get_resources_count() const
{
    return get_db_element()->get_resources_count();
}

const mi::neuraylib::IValue_resource* Module_impl::get_resource( mi::Size index) const
{
    mi::base::Handle<const MDL::IValue_resource> int_resource(
        get_db_element()->get_resource( index));

    mi::base::Handle<Value_factory> vf( get_transaction()->get_value_factory());
    return vf->create<mi::neuraylib::IValue_resource>(
        int_resource.get(), this->cast_to_major());
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

bool Module_impl::is_valid(mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    return get_db_element()->is_valid(
        get_db_transaction(),
        unwrap_and_clear_context(context, default_context));
}

mi::Sint32 Module_impl::reload(
    bool recursive,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* ctx = unwrap_and_clear_context( context, default_context);

    mi::Sint32 result = get_db_element()->reload(
        get_db_transaction(), recursive, ctx);

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Module_impl::reload_from_string(
    const char* module_source,
    bool recursive,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* ctx = unwrap_and_clear_context( context, default_context);
    if( !module_source || strlen( module_source) == 0) {
        MDL::add_error_message( ctx, "Module source cannot be empty.", -1);
        return -1;
    }

    mi::base::Handle<mi::neuraylib::IReader> reader(
        DISK::create_reader( module_source, strlen( module_source)));
    mi::Sint32 result = get_db_element()->reload_from_string(
        get_db_transaction(),
        reader.get(),
        recursive,
        ctx);

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

const mi::neuraylib::IType_resource* Module_impl::deprecated_get_resource_type(
    mi::Size index) const
{
    mi::base::Handle<const mi::neuraylib::IValue_resource> resource( get_resource( index));
    if( !resource)
        return nullptr;

    return resource->get_type();
}

const char* Module_impl::deprecated_get_resource_mdl_file_path( mi::Size index) const
{
    const MDL::Resource_tag_tuple* rtt = get_db_element()->get_resource_tag_tuple( index);
    if( !rtt)
        return nullptr;

    return rtt->m_mdl_file_path.c_str();
}

const char* Module_impl::deprecated_get_resource_name( mi::Size index) const
{
    mi::base::Handle<const mi::neuraylib::IValue_resource> resource( get_resource( index));
    if( !resource)
        return nullptr;

    return resource->get_value();
}

} // namespace NEURAY

} // namespace MI
