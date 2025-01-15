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
 ** \brief Source for the IFunction_call implementation.
 **/

#include "pch.h"

#include <mi/base/handle.h>

#include "neuray_compiled_material_impl.h"
#include "neuray_function_call_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_material_instance_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_impl.h"

#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/inumber.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Function_call_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new MDL::Mdl_function_call;
}

mi::base::IInterface* Function_call_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Function_call_impl())->cast_to_major();
}

const mi::base::IInterface* Function_call_impl::get_interface(
    const mi::base::Uuid& interface_id) const
{
    // Handle all other cases first. In particular, IDb_element is requested before the instance is
    // fully set up and functionality on the underlying DB element, like is_material(), is
    // available.
    if( interface_id != mi::neuraylib::IMaterial_instance::IID())
        return Parent_type::get_interface( interface_id);

    // Handle special case for material instances.
    bool supports_mi = is_material();

    if( interface_id == mi::neuraylib::IMaterial_instance::IID() && !supports_mi)
        return nullptr;
    if( interface_id == mi::neuraylib::IMaterial_instance::IID() &&  supports_mi)
        return Material_instance_impl::create_api_class( this);

    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

mi::base::IInterface* Function_call_impl::get_interface(
    const mi::base::Uuid& interface_id)
{
    // Handle all other cases first. In particular, IDb_element is requested before the instance is
    // fully set up and functionality on the underlying DB element, like is_material(), is
    // available.
    if( interface_id != mi::neuraylib::IMaterial_instance::IID())
        return Parent_type::get_interface( interface_id);

    // Handle special case for material instances.
    //
    // Explicitly call the method is_material() instead of inlining it to ensure that the const
    // overload of get_db_element() is called. This is a workaround for the Python binding which is
    // not correctly tracking the const property of interface pointers.
    bool supports_mi = is_material();

    if( interface_id == mi::neuraylib::IMaterial_instance::IID() && !supports_mi)
        return nullptr;
    if( interface_id == mi::neuraylib::IMaterial_instance::IID() &&  supports_mi)
        return Material_instance_impl::create_api_class( this);

    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

mi::neuraylib::Element_type Function_call_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_FUNCTION_CALL;
}

const char* Function_call_impl::get_function_definition() const
{
    DB::Transaction* db_transaction = get_db_transaction();
    DB::Tag tag = get_db_element()->get_function_definition(db_transaction);
    if (!tag.is_valid())
        return nullptr;
    return db_transaction->tag_to_name( tag);
}

const char* Function_call_impl::get_mdl_function_definition() const
{
    return get_db_element()->get_mdl_function_definition();
}

bool Function_call_impl::is_declarative() const
{
    return get_db_element()->is_declarative();
}

bool Function_call_impl::is_material() const
{
    return get_db_element()->is_material();
}

const mi::neuraylib::IType* Function_call_impl::get_return_type() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IType> result_int( get_db_element()->get_return_type());
    return tf->create( result_int.get(), this->cast_to_major());
}

mi::Size Function_call_impl::get_parameter_count() const
{
    return get_db_element()->get_parameter_count();
}

const char* Function_call_impl::get_parameter_name( mi::Size index) const
{
    return get_db_element()->get_parameter_name( index);
}

mi::Size Function_call_impl::get_parameter_index( const char* name) const
{
    return get_db_element()->get_parameter_index( name);
}

const mi::neuraylib::IType_list* Function_call_impl::get_parameter_types() const
{
    mi::base::Handle<Type_factory> tf( get_transaction()->get_type_factory());
    mi::base::Handle<const MDL::IType_list> result_int( get_db_element()->get_parameter_types());
    return tf->create_type_list( result_int.get(), this->cast_to_major());
}

const mi::neuraylib::IExpression_list* Function_call_impl::get_arguments() const
{
    mi::base::Handle<Expression_factory> ef( get_transaction()->get_expression_factory());
    mi::base::Handle<const MDL::IExpression_list> result_int( get_db_element()->get_arguments());
    return ef->create_expression_list( result_int.get(), this->cast_to_major());
}

mi::Sint32 Function_call_impl::set_arguments( const mi::neuraylib::IExpression_list* arguments)
{
   if( !arguments)
        return -1;

    mi::base::Handle<const MDL::IExpression_list> arguments_int(
        get_internal_expression_list( arguments));

    DB::Tag_set tags;
    MDL::collect_references( arguments_int.get(), &tags);
    for( auto tag : tags)
        if( !can_reference_tag( tag))
            return -7;

    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return get_db_element()->set_arguments( get_db_transaction(), arguments_int.get());
}

mi::Sint32 Function_call_impl::set_argument(
    mi::Size index, const mi::neuraylib::IExpression* argument)
{
    if( !argument)
        return -1;

    mi::base::Handle<const MDL::IExpression> argument_int(
        get_internal_expression( argument));

    DB::Tag_set tags;
    MDL::collect_references( argument_int.get(), &tags);
    for( auto tag : tags)
        if( !can_reference_tag( tag))
            return -7;

    mi::Sint32 result = get_db_element()->set_argument(
        get_db_transaction(), index, argument_int.get());
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Function_call_impl::set_argument(
    const char* name, const mi::neuraylib::IExpression* argument)
{
    if( !argument)
        return -1;

    mi::base::Handle<const MDL::IExpression> argument_int(
        get_internal_expression( argument));

    DB::Tag_set tags;
    MDL::collect_references( argument_int.get(), &tags);
    for( auto tag : tags)
        if( !can_reference_tag( tag))
            return -7;

    mi::Sint32 result = get_db_element()->set_argument(
        get_db_transaction(), name, argument_int.get());
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Function_call_impl::reset_argument( mi::Size index)
{
    mi::Sint32 result = get_db_element()->reset_argument(
        get_db_transaction(), index);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Function_call_impl::reset_argument( const char* name)
{
    mi::Sint32 result = get_db_element()->reset_argument(
        get_db_transaction(), name);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

bool Function_call_impl::is_default() const
{
    return get_db_element()->is_immutable();
}

bool Function_call_impl::is_valid(mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);

    return get_db_element()->is_valid( get_db_transaction(), mdl_context);
}

mi::Sint32 Function_call_impl::repair(
    mi::Uint32 flags, mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);

    mi::Sint32 r = get_db_element()->repair(
        get_db_transaction(),
        flags & mi::neuraylib::MDL_REPAIR_INVALID_ARGUMENTS,
        flags & mi::neuraylib::MDL_REMOVE_INVALID_ARGUMENTS,
        /*level*/ 0,
        mdl_context);

    if( r == 0)
        add_journal_flag(SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return r;
}

mi::neuraylib::ICompiled_material* Function_call_impl::create_compiled_material(
    mi::Uint32 flags, mi::neuraylib::IMdl_execution_context* context) const
{
    if( get_db_element()->is_immutable())
        return nullptr;

    MDL::Execution_context default_context;
    MDL::Execution_context* mdl_context = unwrap_and_clear_context( context, default_context);

    mi::base::Handle<const MDL::IType_struct> target_type_int;
    mi::base::Handle<const mi::base::IInterface> target_type(
        mdl_context->get_interface_option<const mi::base::IInterface>( MDL_CTX_OPTION_TARGET_TYPE));
    if( target_type) {
        mi::base::Handle<const mi::neuraylib::IType> target_type_itype_struct(
            target_type->get_interface<mi::neuraylib::IType_struct>());
        ASSERT( M_NEURAY_API, target_type_itype_struct); // enforced by option validator
        target_type_int = get_internal_type<MDL::IType_struct>( target_type_itype_struct.get());
    }

    bool class_compilation = flags & mi::neuraylib::IMaterial_instance::CLASS_COMPILATION;
    std::shared_ptr<MDL::Mdl_compiled_material> db_instance(
        get_db_element()->create_compiled_material(
            get_db_transaction(), class_compilation, target_type_int.get(), mdl_context));
    if( !db_instance)
        return nullptr;

    auto* api_instance = get_transaction()->create<mi::neuraylib::ICompiled_material>(
        "__Compiled_material");
    static_cast<Compiled_material_impl*>( api_instance)->get_db_element()->swap(
        *db_instance);
    return api_instance;
}

} // namespace NEURAY

} // namespace MI
