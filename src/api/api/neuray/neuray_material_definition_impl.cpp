/***************************************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the Material_definition implementation.
 **/

#include "pch.h"

#include "neuray_material_definition_impl.h"

#include <mi/neuraylib/ifunction_call.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>

#include "neuray_function_definition_impl.h"

namespace MI {

namespace NEURAY {

mi::neuraylib::IMaterial_definition* Material_definition_impl::create_api_class(
    mi::neuraylib::IFunction_definition* impl)
{
    return new Material_definition_impl( impl);
}

const mi::neuraylib::IMaterial_definition* Material_definition_impl::create_api_class(
    const mi::neuraylib::IFunction_definition* impl)
{
    return new Material_definition_impl( const_cast<mi::neuraylib::IFunction_definition*>( impl));
}

const mi::base::IInterface* Material_definition_impl::get_interface(
    const mi::base::Uuid& interface_id) const
{
    return m_impl->get_interface( interface_id);
}

mi::base::IInterface* Material_definition_impl::get_interface(
    const mi::base::Uuid& interface_id)
{
    return m_impl->get_interface( interface_id);
}

mi::IData* Material_definition_impl::create_attribute( const char* name, const char* type)
{
    return m_impl->create_attribute( name, type);
}

bool Material_definition_impl::destroy_attribute( const char* name)
{
    return m_impl->destroy_attribute( name);
}

const mi::IData* Material_definition_impl::access_attribute( const char* name) const
{
    return m_impl->access_attribute( name);
}

mi::IData* Material_definition_impl::edit_attribute( const char* name)
{
    return m_impl->edit_attribute( name);
}

bool Material_definition_impl::is_attribute( const char* name) const
{
    return m_impl->is_attribute( name);
}

const char* Material_definition_impl::get_attribute_type_name( const char* name) const
{
    return m_impl->get_attribute_type_name( name);
}

mi::Sint32 Material_definition_impl::set_attribute_propagation(
    const char* name, mi::neuraylib::Propagation_type value)
{
    return m_impl->set_attribute_propagation( name, value);
}

mi::neuraylib::Propagation_type Material_definition_impl::get_attribute_propagation(
    const char* name) const
{
    return m_impl->get_attribute_propagation( name);
}

const char* Material_definition_impl::enumerate_attributes( mi::Sint32 index) const
{
    return m_impl->enumerate_attributes( index);
}

mi::neuraylib::Element_type Material_definition_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_MATERIAL_DEFINITION;
}

const char* Material_definition_impl::get_module() const
{
    return m_impl->get_module();
}

const char* Material_definition_impl::get_mdl_name() const
{
    return m_impl->get_mdl_name();
}

const char* Material_definition_impl::get_mdl_module_name() const
{
    return m_impl->get_mdl_module_name();
}

const char*  Material_definition_impl::get_mdl_simple_name() const
{
    return m_impl->get_mdl_simple_name();
}

const char* Material_definition_impl::get_mdl_parameter_type_name( mi::Size index) const
{
    return m_impl->get_mdl_parameter_type_name( index);
}

const char* Material_definition_impl::get_prototype() const
{
    return m_impl->get_prototype();
}

void Material_definition_impl::get_mdl_version(
    mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const
{
    return m_impl->get_mdl_version( since, removed);
}

mi::neuraylib::IFunction_definition::Semantics
Material_definition_impl::get_semantic() const
{
    return m_impl->get_semantic();
}

bool Material_definition_impl::is_exported() const
{
    return m_impl->is_exported();
}

const mi::neuraylib::IType* Material_definition_impl::get_return_type() const
{
    return m_impl->get_return_type();
}

mi::Size Material_definition_impl::get_parameter_count() const
{
    return m_impl->get_parameter_count();
}

const char* Material_definition_impl::get_parameter_name( mi::Size index) const
{
    return m_impl->get_parameter_name( index);
}

mi::Size Material_definition_impl::get_parameter_index( const char* name) const
{
    return m_impl->get_parameter_index( name);
}

const mi::neuraylib::IType_list* Material_definition_impl::get_parameter_types() const
{
    return m_impl->get_parameter_types();
}

const mi::neuraylib::IExpression_list* Material_definition_impl::get_defaults() const
{
    return m_impl->get_defaults();
}

const mi::neuraylib::IExpression_list* Material_definition_impl::get_enable_if_conditions() const
{
    return m_impl->get_enable_if_conditions();
}

mi::Size Material_definition_impl::get_enable_if_users( mi::Size index) const
{
    return m_impl->get_enable_if_users( index);
}

mi::Size Material_definition_impl::get_enable_if_user( mi::Size index, mi::Size u_index) const
{
    return m_impl->get_enable_if_user( index, u_index);
}

const mi::neuraylib::IAnnotation_block* Material_definition_impl::get_annotations() const
{
    return m_impl->get_annotations();
}

const mi::neuraylib::IAnnotation_block* Material_definition_impl::get_return_annotations() const
{
    return m_impl->get_return_annotations();
}

const mi::neuraylib::IAnnotation_list* Material_definition_impl::get_parameter_annotations() const
{
    return m_impl->get_parameter_annotations();
}

const char* Material_definition_impl::get_thumbnail() const
{
    return m_impl->get_thumbnail();
}

bool Material_definition_impl::is_valid( mi::neuraylib::IMdl_execution_context* context) const
{
    return m_impl->is_valid( context);
}

const mi::neuraylib::IExpression_direct_call* Material_definition_impl::get_body() const
{
    mi::base::Handle<const mi::neuraylib::IExpression> body( m_impl->get_body());
    return body ? body->get_interface<mi::neuraylib::IExpression_direct_call>() : nullptr;
}

mi::Size Material_definition_impl::get_temporary_count() const
{
    return m_impl->get_temporary_count();
}

const mi::neuraylib::IExpression* Material_definition_impl::get_temporary( mi::Size index) const
{
    return m_impl->get_temporary( index);
}

const char* Material_definition_impl::get_temporary_name( mi::Size index) const
{
    return m_impl->get_temporary_name( index );
}

mi::neuraylib::IMaterial_instance* Material_definition_impl::create_material_instance(
    const mi::neuraylib::IExpression_list* arguments, mi::Sint32* errors) const
{
    mi::base::Handle<mi::neuraylib::IFunction_call> fc(
        m_impl->create_function_call( arguments, errors));
    return fc ? fc->get_interface<mi::neuraylib::IMaterial_instance>() : nullptr;
}

const MDL::Mdl_function_definition* Material_definition_impl::get_db_element() const
{
    const Function_definition_impl* fd = static_cast<const Function_definition_impl*>( m_impl.get());
    return static_cast<const MDL::Mdl_function_definition*>( fd->get_db_element());
}

MDL::Mdl_function_definition* Material_definition_impl::get_db_element()
{
    Function_definition_impl* fd = static_cast<Function_definition_impl*>( m_impl.get());
    return static_cast<MDL::Mdl_function_definition*>( fd->get_db_element());
}

Material_definition_impl::Material_definition_impl( mi::neuraylib::IFunction_definition* impl)
  : m_impl( impl, mi::base::DUP_INTERFACE)
{
}

} // namespace NEURAY

} // namespace MI
