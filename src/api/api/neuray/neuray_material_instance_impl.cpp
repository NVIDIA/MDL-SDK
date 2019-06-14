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
 ** \brief Source for the IMaterial_instance implementation.
 **/

#include "pch.h"

#include "neuray_material_instance_impl.h"

#include <mi/base/handle.h>
#include <boost/shared_ptr.hpp>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>

#include "neuray_compiled_material_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"

namespace MI {

namespace NEURAY {

DB::Element_base* Material_instance_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return 0;
    return new MDL::Mdl_material_instance;
}

mi::base::IInterface* Material_instance_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return 0;
    return (new Material_instance_impl())->cast_to_major();
}

mi::neuraylib::Element_type Material_instance_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_MATERIAL_INSTANCE;
}

const char* Material_instance_impl::get_material_definition() const
{
    DB::Tag tag = get_db_element()->get_material_definition();
    return get_db_transaction()->tag_to_name( tag);
}

const char* Material_instance_impl::get_mdl_material_definition() const
{
    return get_db_element()->get_mdl_material_definition();
}

mi::Size Material_instance_impl::get_parameter_count() const
{
    return get_db_element()->get_parameter_count();
}

const char* Material_instance_impl::get_parameter_name( mi::Size index) const
{
    return get_db_element()->get_parameter_name( index);
}

mi::Size Material_instance_impl::get_parameter_index( const char* name) const
{
    return get_db_element()->get_parameter_index( name);
}

const mi::neuraylib::IType_list* Material_instance_impl::get_parameter_types() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IType_list> result_int( get_db_element()->get_parameter_types());
    return tf->create_type_list( result_int.get(), this->cast_to_major());
}

const mi::neuraylib::IExpression_list* Material_instance_impl::get_arguments() const
{
    mi::base::Handle<Expression_factory> ef( get_transaction()->get_expression_factory());
    mi::base::Handle<const MDL::IExpression_list> result_int( get_db_element()->get_arguments());
    return ef->create_expression_list( result_int.get(), this->cast_to_major());
}

mi::Sint32 Material_instance_impl::set_arguments(
    const mi::neuraylib::IExpression_list* arguments)
{
    if( !arguments)
        return -1;

    mi::base::Handle<const MDL::IExpression_list> arguments_int(
        get_internal_expression_list( arguments));

    DB::Tag_set tags;
    MDL::collect_references( arguments_int.get(), &tags);
    for( DB::Tag_set::const_iterator it = tags.begin(); it != tags.end(); ++it)
        if( !can_reference_tag( *it))
            return -7;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return get_db_element()->set_arguments( get_db_transaction(), arguments_int.get());
}

mi::Sint32 Material_instance_impl::set_argument(
    mi::Size index, const mi::neuraylib::IExpression* argument)
{
    if( !argument)
        return -1;
    mi::base::Handle<const MDL::IExpression> argument_int( get_internal_expression( argument));

    DB::Tag_set tags;
    MDL::collect_references( argument_int.get(), &tags);
    for( DB::Tag_set::const_iterator it = tags.begin(); it != tags.end(); ++it)
        if( !can_reference_tag( *it))
            return -7;

    mi::Sint32 result = get_db_element()->set_argument(
        get_db_transaction(), index, argument_int.get());
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Material_instance_impl::set_argument(
    const char* name, const mi::neuraylib::IExpression* argument)
{
    if( !argument)
        return -1;

    mi::base::Handle<const MDL::IExpression> argument_int( get_internal_expression( argument));

    DB::Tag_set tags;
    MDL::collect_references( argument_int.get(), &tags);
    for( DB::Tag_set::const_iterator it = tags.begin(); it != tags.end(); ++it)
        if( !can_reference_tag( *it))
            return -7;

    mi::Sint32 result = get_db_element()->set_argument(
        get_db_transaction(), name, argument_int.get());
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::neuraylib::ICompiled_material* Material_instance_impl::deprecated_create_compiled_material(
    mi::Uint32 flags,
    mi::Float32 mdl_meters_per_scene_unit,
    mi::Float32 mdl_wavelength_min,
    mi::Float32 mdl_wavelength_max,
    mi::Sint32* errors) const
{
    Mdl_execution_context_impl context;
    context.set_option(MDL_CTX_OPTION_METERS_PER_SCENE_UNIT, mdl_meters_per_scene_unit);
    context.set_option(MDL_CTX_OPTION_WAVELENGTH_MIN, mdl_wavelength_min);
    context.set_option(MDL_CTX_OPTION_WAVELENGTH_MAX, mdl_wavelength_max);
    
    mi::neuraylib::ICompiled_material* cm = create_compiled_material(flags, &context);
    if (errors)
        *errors = context.get_context().get_result();

    return cm;
}

mi::neuraylib::ICompiled_material* Material_instance_impl::create_compiled_material(
    mi::Uint32 flags,
    mi::neuraylib::IMdl_execution_context* context) const
{
    if (get_db_element()->is_immutable())
        return nullptr;

    MDL::Execution_context default_context;
    NEURAY::Mdl_execution_context_impl* context_impl =
        static_cast<NEURAY::Mdl_execution_context_impl*>(context);

    bool class_compilation = flags & CLASS_COMPILATION;
    boost::shared_ptr<MDL::Mdl_compiled_material> db_instance(
        get_db_element()->create_compiled_material(
            get_db_transaction(), class_compilation,
            context_impl ? &context_impl->get_context() : &default_context));

    if (!db_instance)
        return 0;
    mi::neuraylib::ICompiled_material* api_instance
        = get_transaction()->create<mi::neuraylib::ICompiled_material>(
            "__Compiled_material");
    static_cast<Compiled_material_impl*>(api_instance)->get_db_element()->swap(
        *db_instance.get());
    return api_instance;
}

bool Material_instance_impl::is_default() const
{
    return get_db_element()->is_immutable();
}

} // namespace NEURAY

} // namespace MI
