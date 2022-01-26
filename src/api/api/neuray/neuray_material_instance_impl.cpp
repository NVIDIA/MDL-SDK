/***************************************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/neuraylib/ifunction_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>

#include "neuray_function_call_impl.h"

namespace MI {

namespace NEURAY {

mi::neuraylib::IMaterial_instance* Material_instance_impl::create_api_class(
    mi::neuraylib::IFunction_call* impl)
{
    return new Material_instance_impl( impl);
}

const mi::neuraylib::IMaterial_instance* Material_instance_impl::create_api_class(
    const mi::neuraylib::IFunction_call* impl)
{
    return new Material_instance_impl( const_cast<mi::neuraylib::IFunction_call*>( impl));
}

mi::base::IInterface* Material_instance_impl::get_interface(
    const mi::base::Uuid& interface_id)
{
    return m_impl->get_interface( interface_id);
}

const mi::base::IInterface* Material_instance_impl::get_interface(
    const mi::base::Uuid& interface_id) const
{
    return m_impl->get_interface( interface_id);
}

mi::IData* Material_instance_impl::create_attribute( const char* name, const char* type)
{
    return m_impl->create_attribute( name, type);
}

bool Material_instance_impl::destroy_attribute( const char* name)
{
    return m_impl->destroy_attribute( name);
}

const mi::IData* Material_instance_impl::access_attribute( const char* name) const
{
    return m_impl->access_attribute( name);
}

mi::IData* Material_instance_impl::edit_attribute( const char* name)
{
    return m_impl->edit_attribute( name);
}

bool Material_instance_impl::is_attribute( const char* name) const
{
    return m_impl->is_attribute( name);
}

const char* Material_instance_impl::get_attribute_type_name( const char* name) const
{
    return m_impl->get_attribute_type_name( name);
}

mi::Sint32 Material_instance_impl::set_attribute_propagation(
    const char* name, mi::neuraylib::Propagation_type value)
{
    return m_impl->set_attribute_propagation( name, value);
}

mi::neuraylib::Propagation_type Material_instance_impl::get_attribute_propagation(
    const char* name) const
{
    return m_impl->get_attribute_propagation( name);
}

const char* Material_instance_impl::enumerate_attributes( mi::Sint32 index) const
{
    return m_impl->enumerate_attributes( index);
}

mi::neuraylib::Element_type Material_instance_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_MATERIAL_INSTANCE;
}

const char* Material_instance_impl::get_material_definition() const
{
    return m_impl->get_function_definition();
}

const char* Material_instance_impl::get_mdl_material_definition() const
{
    return m_impl->get_mdl_function_definition();
}

const mi::neuraylib::IType* Material_instance_impl::get_return_type() const
{
    return m_impl->get_return_type();
}

mi::Size Material_instance_impl::get_parameter_count() const
{
    return m_impl->get_parameter_count();
}

const char* Material_instance_impl::get_parameter_name( mi::Size index) const
{
    return m_impl->get_parameter_name( index);
}

mi::Size Material_instance_impl::get_parameter_index( const char* name) const
{
    return m_impl->get_parameter_index( name);
}

const mi::neuraylib::IType_list* Material_instance_impl::get_parameter_types() const
{
    return m_impl->get_parameter_types();
}

const mi::neuraylib::IExpression_list* Material_instance_impl::get_arguments() const
{
    return m_impl->get_arguments();
}

mi::Sint32 Material_instance_impl::set_arguments(
    const mi::neuraylib::IExpression_list* arguments)
{
    return m_impl->set_arguments( arguments);
}

mi::Sint32 Material_instance_impl::set_argument(
    mi::Size index, const mi::neuraylib::IExpression* argument)
{
    return m_impl->set_argument( index, argument);
}

mi::Sint32 Material_instance_impl::set_argument(
    const char* name, const mi::neuraylib::IExpression* argument)
{
    return m_impl->set_argument( name, argument);
}

mi::Sint32 Material_instance_impl::reset_argument( mi::Size index)
{
    return m_impl->reset_argument( index);
}

mi::Sint32 Material_instance_impl::reset_argument( const char* name)
{
    return m_impl->reset_argument( name);
}

bool Material_instance_impl::is_default() const
{
    return m_impl->is_default();
}

bool Material_instance_impl::is_valid( mi::neuraylib::IMdl_execution_context* context) const
{
    return m_impl->is_valid( context);
}

mi::Sint32 Material_instance_impl::repair(
    mi::Uint32 flags,
    mi::neuraylib::IMdl_execution_context* context)
{
    return m_impl->repair( flags, context);
}

mi::neuraylib::ICompiled_material* Material_instance_impl::create_compiled_material(
    mi::Uint32 flags,
    mi::neuraylib::IMdl_execution_context* context) const
{
    const Function_call_impl* fc = static_cast<const Function_call_impl*>( m_impl.get());
    return fc->create_compiled_material( flags, context);
}

const MDL::Mdl_function_call* Material_instance_impl::get_db_element() const
{
    const Function_call_impl* fc = static_cast<const Function_call_impl*>( m_impl.get());
    return static_cast<const MDL::Mdl_function_call*>( fc->get_db_element());
}

MDL::Mdl_function_call* Material_instance_impl::get_db_element()
{
    Function_call_impl* fc = static_cast<Function_call_impl*>( m_impl.get());
    return static_cast<MDL::Mdl_function_call*>( fc->get_db_element());
}

Material_instance_impl::Material_instance_impl( mi::neuraylib::IFunction_call* impl)
  : m_impl( impl, mi::base::DUP_INTERFACE)
{
}

} // namespace NEURAY

} // namespace MI
